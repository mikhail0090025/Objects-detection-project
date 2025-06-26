import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
# from get_dataset import input_images, spots_of_interest, max_objects, categories_count
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from my_dataset_handler import all_images, start_size, initial_size, max_objects
import math
import cv2

def compute_iou(anchors, gt_boxes):
    N = anchors.size(0)
    batch_size = gt_boxes.size(0)
    num_boxes = gt_boxes.size(1)  # 6
    iou = torch.zeros(N, num_boxes * batch_size).to(anchors.device)
    
    for i in range(batch_size):
        gt_coords = gt_boxes[i, :, :4]  # Берем только [x1, y1, x2, y2], форма [6, 4]
        print("Shape 1: ", anchors[:, 0].unsqueeze(1).shape, gt_coords[:, 0].unsqueeze(0).shape)
        print("Shape 2: ", anchors[:, 0].shape, gt_coords[:, 0].shape)
        print("Shape 3: ", anchors.shape, gt_coords.shape)
        print("Shape 4: ", gt_boxes.shape)
        print("--------------------------------")
        x1 = torch.max(anchors[:, 0].unsqueeze(1), gt_coords[:, 0].unsqueeze(0))  # [1764, 6]
        y1 = torch.max(anchors[:, 1].unsqueeze(1), gt_coords[:, 1].unsqueeze(0))
        x2 = torch.min(anchors[:, 2].unsqueeze(1), gt_coords[:, 2].unsqueeze(0))
        y2 = torch.min(anchors[:, 3].unsqueeze(1), gt_coords[:, 3].unsqueeze(0))
        
        inter_width = torch.clamp(x2 - x1, min=0)
        inter_height = torch.clamp(y2 - y1, min=0)
        inter_area = inter_width * inter_height
        
        anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        gt_area = (gt_coords[:, 2] - gt_coords[:, 0]) * (gt_coords[:, 3] - gt_coords[:, 1])
        
        union_area = anchor_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area
        iou[:, i * num_boxes:(i + 1) * num_boxes] = inter_area / (union_area + 1e-6)
    
    return iou

def compute_reg_targets(anchors, gt_boxes):
    # Вычисляем [dx, dy, dw, dh]
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + anchor_w / 2
    anchor_cy = anchors[:, 1] + anchor_h / 2
    
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + gt_w / 2
    gt_cy = gt_boxes[:, 1] + gt_h / 2
    
    dx = (gt_cx - anchor_cx) / anchor_w
    dy = (gt_cy - anchor_cy) / anchor_h
    dw = torch.log(gt_w / anchor_w)
    dh = torch.log(gt_h / anchor_h)
    
    return torch.stack([dx, dy, dw, dh], dim=1)

def prepare_rpn_targets(anchors, annotations):
    ious = compute_iou(anchors, annotations)
    max_ious, gt_indices = ious.max(dim=1)
    
    targets_cls = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
    targets_cls[max_ious >= 0.7] = 1  # Объект
    targets_cls[max_ious < 0.3] = 0   # Фон
    mask = (max_ious >= 0.7)  # Учитываем только объекты для регрессии
    
    targets_reg = torch.zeros(len(anchors), 4, device=anchors.device)
    if mask.sum() > 0:
        targets_reg[mask] = compute_reg_targets(anchors[mask], annotations[gt_indices[mask]])
    
    return targets_cls, targets_reg, mask

def prepare_cls_targets(filtered_anchors, annotations, batch_index=0):
    # Берем аннотации только для одного изображения (batch_index)
    annotations_img = annotations[batch_index]  # [7, 6]
    
    # Фильтруем валидные объекты (предполагаем, что -1 в class_id означает "нет объекта")
    valid_anno = annotations_img[annotations_img[:, 4] != -1]  # [M, 6], где M — число реальных объектов
    
    if valid_anno.size(0) == 0:
        # Если нет объектов, все якоря — фон
        targets_cls = torch.zeros(len(filtered_anchors), dtype=torch.long, device=filtered_anchors.device)
        targets_bbox = torch.zeros(len(filtered_anchors), 4, device=filtered_anchors.device)
        mask = torch.zeros(len(filtered_anchors), dtype=torch.bool, device=filtered_anchors.device)
        return targets_cls, targets_bbox, mask
    
    # Вычисляем IoU между якорями и ground truth боксами
    ious = compute_iou(filtered_anchors, valid_anno[:, :4])  # [N, M], где N — число якорей
    max_ious, gt_indices = ious.max(dim=1)  # [N], [N]
    
    # Инициализируем целевые классы
    targets_cls = torch.zeros(len(filtered_anchors), dtype=torch.long, device=filtered_anchors.device)
    mask = max_ious > 0.5  # Порог для классификации
    
    if mask.sum() > 0:
        # Присваиваем классы для якорей с IoU > 0.5
        targets_cls[mask] = valid_anno[gt_indices[mask], 4].long()
    targets_cls[~mask] = 0  # Фон
    
    # Инициализируем целевые координаты
    targets_bbox = torch.zeros(len(filtered_anchors), 4, device=filtered_anchors.device)
    if mask.sum() > 0:
        targets_bbox[mask] = compute_reg_targets(filtered_anchors[mask], valid_anno[gt_indices[mask], :4])
    
    return targets_cls, targets_bbox, mask

def get_selective_search_proposals(image):
    # Преобразуем изображение в RGB (OpenCV использует BGR по умолчанию)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Инициализируем Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    
    # Настраиваем стратегию (можно менять)
    ss.switchToSelectiveSearchQuality()  # Или switchToSelectiveSearchFast() для скорости
    
    # Получаем регионы
    rects = ss.process()
    
    # Ограничиваем число регионов (например, 2000)
    rects = rects[:2000]
    
    # Преобразуем в список координат [x1, y1, x2, y2]
    proposals = []
    for rect in rects:
        x, y, w, h = rect
        x2, y2 = x + w, y + h
        proposals.append([x, y, x2, y2])
    
    return np.array(proposals)

def generate_anchors(feature_map_size, img_size, scales=[128, 256, 512], ratios=[0.5, 1, 2]):
    # feature_map_size: [H, W] размера feature map
    # img_size: [H, W] исходного изображения
    # scales и ratios для создания якорей
    print(feature_map_size)
    print(img_size)
    anchors = []
    width_fm, height_fm = feature_map_size
    img_width, img_height = img_size
    strideX = img_width / width_fm
    strideY = img_height / height_fm
    
    for h in range(height_fm):
        for w in range(width_fm):
            for scale in scales:
                for ratio in ratios:
                    # Центр якора в пространстве изображения
                    center_h = (h + 0.5) * strideX
                    center_w = (w + 0.5) * strideY
                    
                    # Размеры якорей в пикселях изображения
                    h_ = math.sqrt(scale * scale / ratio)
                    w_ = h_ * ratio
                    
                    # Координаты [x1, y1, x2, y2] в пикселях изображения
                    x1 = max(0, center_w - w_ / 2)
                    y1 = max(0, center_h - h_ / 2)
                    x2 = min(img_width, center_w + w_ / 2)
                    y2 = min(img_height, center_h + h_ / 2)
                    
                    anchors.append([x1, y1, x2, y2])
    
    return torch.tensor(anchors, dtype=torch.float32)

class SpotsOfInterestDataset(Dataset):
    def __init__(self, all_images):
        for img in all_images:
            img.image = img.image + 1
            img.image = img.image * 127.5
            img.image = img.image.astype(np.uint8)
            Image.fromarray(img.image).save("tmp.jpg")
            img.image = cv2.imread("tmp.jpg")
        self.inputs = np.array([img.image for img in all_images])
        self.outputs = np.array([img.output for img in all_images])
        # Преобразуем изображения
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        annotation = self.outputs[idx]
        return image, annotation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential() if stride == 1 and in_channels == out_channels else \
                          nn.Sequential(
                              nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                              nn.BatchNorm2d(out_channels)
                          )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class FeatureMapNet(nn.Module):
    def __init__(self):
        super(FeatureMapNet, self).__init__()
        from torchvision.models import vgg16, vgg16_bn, vgg11_bn
        self.vgg = vgg11_bn(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg(x)
        return x

class MyBackbone(nn.Module):
    def __init__(self, output_size=(14,14)):
        super(MyBackbone, self).__init__()
        self.all_layers = nn.Sequential(
            ResidualBlock(3, 32, 1), # x1
            ResidualBlock(32, 32, 2), # x2
            ResidualBlock(32, 64, 2), # x4
            ResidualBlock(64, 128, 2), # x8
            ResidualBlock(128, 256, 2), # x16
            ResidualBlock(256, 512, 2), # x32
            nn.AdaptiveAvgPool2d(output_size),
        )

    def forward(self, x):
        x = self.all_layers(x)
        return x

class RoI(nn.Module):
    def __init__(self):
        super(RoI, self).__init__()
        self.main_layer = nn.AdaptiveMaxPool2d(7)

    def forward(self, x):
        x = self.main_layer(x)
        return x

feat_map_channels = 512  # Каналы feature map (например, из VGG16)
classes = 2
anchors_scales = [128, 256, 512]
anchors_ratios = [0.5, 1, 2]
num_anchors = len(anchors_scales) * len(anchors_ratios)
classes_ = classes

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        
        # Слои RPN
        self.conv1 = nn.Conv2d(feat_map_channels, 512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, kernel_size=1)  # 2 класса: объект/фон
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, kernel_size=1)  # 4 координаты на якорь
        
        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: feature map, например, [batch_size, 512, H, W]
        batch_size = x.size(0)
        
        # Свёрточный слой для извлечения признаков
        x = torch.nn.functional.relu(self.conv1(x))
        
        # Предсказание вероятностей (объект/фон)
        cls = self.cls_layer(x)  # [batch_size, 18, H, W] (9 якорей x 2 класса)
        cls = cls.permute(0, 2, 3, 1).contiguous()  # [batch_size, H, W, 18]
        cls = cls.view(batch_size, -1, 2)  # [batch_size, H*W*9, 2]
        
        # Предсказание координат (регрессия)
        reg = self.reg_layer(x)  # [batch_size, 36, H, W] (9 якорей x 4 координаты)
        reg = reg.permute(0, 2, 3, 1).contiguous()  # [batch_size, H, W, 36]
        reg = reg.view(batch_size, -1, 4)  # [batch_size, H*W*9, 4]
        
        return cls, reg

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=1, stride=7)  # Уменьшили каналы до 1024
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.output = nn.Linear(1024, classes + 1)
        self.softmax = nn.Softmax(dim=-1)  # Явно указываем dim=-1

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.output(x)
        return x

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.feature = MyBackbone()
        self.rpn = RPN()
        from torchvision.ops import RoIAlign
        self.roi_align = RoIAlign((7, 7), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        self.classifier = Classifier()
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        import time
        print("Images count in batch: ", x.shape[0])
        # Получаем feature map
        feature_map = self.feature(x)
        batch_size, channels, width_fm, height_fm = feature_map.shape
        # Генерируем якоря
        anchors = generate_anchors([width_fm, height_fm], [x.shape[2], x.shape[3]], scales=anchors_scales, ratios=anchors_ratios)
        
        # Получаем предсказания от RPN
        cls_scores, reg_coords = self.rpn(feature_map)
        scores = torch.softmax(cls_scores[0], dim=-1)[:, 1]
        _, topk_indices = torch.topk(scores, min(2000, len(scores)))  # Берем топ-2000 регионов
        filtered_anchors = anchors[topk_indices]

        # Подготовка координат для RoIAlign
        batch_indices = torch.zeros(len(filtered_anchors), dtype=torch.long).to(x.device)
        rois_coords = torch.stack([batch_indices, filtered_anchors[:, 0], filtered_anchors[:, 1],
                                filtered_anchors[:, 2], filtered_anchors[:, 3]], dim=1).float()
        
        # Вырезаем регионы с помощью RoIAlign
        rois = self.roi_align(feature_map, rois_coords)
        
        # Классификация регионов
        classes = self.classifier(rois)
        class_probs = torch.softmax(classes, dim=-1)
        
        # Векторизованная фильтрация фоновых регионов
        max_confidence, _ = torch.max(class_probs, dim=-1)  # [num_rois]
        uniform_threshold = 1.0 / (classes_ + 1)  # Явно задаём скаляр (classes + 1 = 8)
        uniform_threshold = 1
        background_mask = (class_probs[:, -1] == max_confidence) | (class_probs[:, -1] > uniform_threshold)
        keep_mask = ~background_mask
        print("Keep mask: ", keep_mask, len([x for x in keep_mask if x == True]))
        
        rois = rois[keep_mask]
        class_probs = class_probs[keep_mask]
        
        bbox = self.bbox_regressor(rois)
        
        print("RoIs shape:", rois.shape)
        
        return cls_scores, reg_coords, classes, bbox, filtered_anchors, anchors, keep_mask

frcnn = FasterRCNN()
frcnn(torch.rand(8, 3, 640, 480))
print('-----')

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        rmse = torch.sqrt(l1 + 1e-7)
        return rmse

class RPNLoss(nn.Module):
    def __init__(self):
        super(RPNLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='none')  # Без суммирования
        self.crossentropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, mask):
        # pred: [reg_coords, cls_scores]
        # target: [targets_reg, targets_cls]
        # mask: булева маска положительных якорей [N]
        
        reg_pred, cls_pred = pred
        reg_target, cls_target = target
        
        # Потеря классификации для всех якорей
        loss_cls = self.crossentropy_loss(cls_pred, cls_target)
        
        # Потеря регрессии только для положительных якорей
        loss_reg = self.l1_loss(reg_pred, reg_target)
        loss_reg = torch.mean(loss_reg[mask]) if mask.sum() > 0 else 0.0  # Среднее по маске
        
        # Балансировка (можно добавить вес lambda)
        return loss_cls.mean() + 10 * loss_reg  # lambda = 10, как в оригинале

class CustomDistanceLoss(nn.Module):
    def __init__(self, max_objects=7):
        super(CustomDistanceLoss, self).__init__()
        self.max_objects = max_objects

    def forward(self, pred_boxes, target_boxes):
        loss = 0
        for pred, target in zip(pred_boxes, target_boxes):
            # valid_pred = pred[pred.sum(dim=1) > 0]  # Фильтруем нулевые
            # valid_target = target[target.sum(dim=1) > 0]
            valid_pred = pred
            valid_target = target
            if len(valid_pred) > 0 and len(valid_target) > 0:
                # Центр бокса для расстояния
                pred_center_x = (valid_pred[:, 0] + valid_pred[:, 2]) / 2
                pred_center_y = (valid_pred[:, 1] + valid_pred[:, 3]) / 2
                target_center_x = (valid_target[:, 0] + valid_target[:, 2]) / 2
                target_center_y = (valid_target[:, 1] + valid_target[:, 3]) / 2
                # Евклидово расстояние
                distance = torch.sqrt((pred_center_x - target_center_x) ** 2 + 
                                    (pred_center_y - target_center_y) ** 2)
                loss += distance.mean()  # Среднее расстояние
        return loss / len(pred_boxes) if len(pred_boxes) > 0 else 0

# Инициализация
train_images, val_images = train_test_split(
    all_images, test_size=0.2, random_state=42
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = SpotsOfInterestDataset(train_images)
val_dataset = SpotsOfInterestDataset(val_images)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
detector = FasterRCNN().to(device)
rpn_loss = RPNLoss()
optimizer = torch.optim.Adam(detector.parameters(), lr=0.001, weight_decay=0.001)
all_losses = []
all_val_losses = []

def rmse_loss(pred, target):
    return torch.sqrt(torch.nn.L1Loss(pred, target) + 1e-7)

def one_epoch():
    for images, annotations in train_dataloader:
        optimizer.zero_grad()
        targets_reg = annotations
        
        # Прямой проход
        cls_scores, reg_coords, classes, bbox, filtered_anchors, anchors, rpn_mask = detector(images)
        
        # Подготовка меток
        targets_cls, targets_bbox, cls_mask = prepare_cls_targets(filtered_anchors, annotations)
        
        # Лосс классификации
        loss_cls = nn.CrossEntropyLoss()(classes[cls_mask], targets_cls[cls_mask])
        
        # Лосс регрессии (и RPN лоссы)
        loss_bbox = nn.SmoothL1Loss()(bbox[cls_mask], targets_bbox[cls_mask])
        loss_rpn = rpn_loss((reg_coords, cls_scores), (targets_reg, targets_cls), rpn_mask)
        
        # Общая потеря
        loss = loss_rpn + loss_cls + loss_bbox
        loss.backward()
        optimizer.step()
    
    print(f"Epoch passed, Loss: {loss.item()}")

# Vizualization
def show_graphics():
    plt.plot(all_losses, color="blue")
    plt.plot(all_val_losses, color="orange")
    plt.grid()
    plt.show()

one_epoch()