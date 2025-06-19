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

class SpotsOfInterestDataset(Dataset):
    def __init__(self, all_images):
        for img in all_images:
            img.image = img.image + 1
            img.image = img.image * 127.5
            img.image = img.image.astype(np.uint8)
            print('img')
            print(img.image)
            print(img.image.min())
            print(img.image.max())
            print(img.image.shape)
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
    def __init__(self, max_objects=7):
        super(FeatureMapNet, self).__init__()
        from torchvision.models import vgg16, vgg16_bn, vgg11_bn
        self.vgg = vgg11_bn(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg(x)
        return x

class FasterRCNN(nn.Module):
    def __init__(self, max_objects=7):
        super(FasterRCNN, self).__init__()
        self.feature = FeatureMapNet()

    def forward(self, x):
        x = self.feature(x)
        return x

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        rmse = torch.sqrt(l1 + 1e-7)
        return rmse

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
                print('--------------------')
                print('valid_pred')
                print(valid_pred)
                print('valid_target')
                print(valid_target)
                print('--------------------')
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
optim_detector = optim.Adam(detector.parameters(), lr=0.0001, weight_decay=0.01)
all_losses = []
all_val_losses = []

def rmse_loss(pred, target):
    return torch.sqrt(torch.nn.L1Loss(pred, target) + 1e-7)

# Использование
# loss_fn = RMSELoss()
# loss_fn = CustomDistanceLoss()
loss_fn = torch.nn.L1Loss()
# Тренировка
def one_epoch():
    detector.train()
    epoch_loss = 0
    val_loss = 0
    for i, (images, targets) in enumerate(train_dataloader):
        images, targets = images.to(device), targets.to(device)
        optim_detector.zero_grad()
        output = detector(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optim_detector.step()
        epoch_loss += loss.item()
        # print(f"Batch {i+1}/{len(train_dataloader)}: Loss: {loss.item():.4f}")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_dataloader):
            images, targets = images.to(device), targets.to(device)
            optim_detector.zero_grad()
            output = detector(images)
            loss = loss_fn(output, targets)
            val_loss += loss.item()
            # print(f"Batch {i+1}/{len(train_dataloader)}: Loss: {loss.item():.4f}")

    epoch_loss = epoch_loss / len(train_dataloader)
    val_loss = val_loss / len(val_dataloader)
    all_losses.append(epoch_loss)
    all_val_losses.append(val_loss)
    return epoch_loss, val_loss

def go_epochs(epochs_count):
    loss, val_loss = (0,0)
    for i in range(epochs_count):
        loss, val_loss = one_epoch()
        print(f"Epoch {i+1}/{epochs_count}, loss: {loss}, val loss: {val_loss}")
    return loss, val_loss

# Vizualization
def show_graphics():
    plt.plot(all_losses, color="blue")
    plt.plot(all_val_losses, color="orange")
    plt.grid()
    plt.show()