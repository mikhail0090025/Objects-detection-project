import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from get_dataset import input_images, outputs, max_objects_per_img, categories_count  # Предполагаем, что эти переменные экспортированы

# Определяем output_shape
output_shape = (max_objects_per_img, 4 + categories_count)

class LearningDataset(Dataset):
    def __init__(self, images, annotations):
        # Преобразуем изображения
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        # Преобразуем аннотации в фиксированный тензор
        self.annotations = torch.zeros((len(annotations), max_objects_per_img, 4 + categories_count), dtype=torch.float32)
        print(annotations.shape)
        for i, (_, anns) in enumerate(annotations):
            for j, ann in enumerate(anns):
                if j >= max_objects_per_img:
                    break
                self.annotations[i, j] = torch.tensor(ann, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        annotation = self.annotations[idx]
        return image, annotation

class ObjectsDetector(nn.Module):
    def __init__(self):
        super(ObjectsDetector, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2),  # 200x200 => 100x100
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 100x100 => 50x50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),  # 50x50 => 25x25
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),  # 25x25 => 13x13
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 16, 3, padding=1, stride=1),  # 13x13 => 13x13
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 4 + categories_count, 3, padding=1),  # 13x13 => 13x13
            nn.BatchNorm2d(4 + categories_count),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(13 * 13 * (4 + categories_count), max_objects_per_img * (4 + categories_count)),
            nn.Unflatten(1, (max_objects_per_img, 4 + categories_count)),
            nn.Sigmoid()  # Нормализация [0, 1] для координат и классов
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class FullDetector(nn.Module):
    def __init__(self):
        super(FullDetector, self).__init__()
        self.objects_detector = ObjectsDetector()

    def forward(self, x):
        return self.objects_detector(x)

# Инициализация
dataset = LearningDataset(input_images, outputs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
detector = FullDetector().to(device)
optim_detector = optim.Adam(detector.parameters(), lr=0.0001)

# Кастомная потеря для детекции
class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Для классов
        self.mse_loss = nn.MSELoss()  # Для координат

    def forward(self, pred, target):
        batch_size = pred.size(0)
        loss = 0
        for i in range(batch_size):
            pred_boxes = pred[i, :, :4]  # Координаты
            target_boxes = target[i, :, :4]
            pred_classes = pred[i, :, 4:]  # Классы
            target_classes = target[i, :, 4:]

            # Маска для существующих объектов
            mask = (target_boxes.sum(dim=1) > 0).float()  # 1, если объект есть
            if mask.sum() == 0:  # Если объектов нет
                continue

            # Потери для координат
            box_loss = self.mse_loss(pred_boxes[mask.bool()], target_boxes[mask.bool()])
            # Потери для классов
            class_loss = self.bce_loss(pred_classes[mask.bool()], target_classes[mask.bool()])
            loss += (box_loss + class_loss) / mask.sum()
        return loss / batch_size

loss_fn = DetectionLoss()

# Тренировка
def one_epoch():
    detector.train()
    epoch_loss = 0
    for i, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        optim_detector.zero_grad()
        output = detector(images)
        loss = loss_fn(output, targets)
        loss.backward()
        optim_detector.step()
        epoch_loss += loss.item()
        print(f"Batch {i+1}/{len(dataloader)}: Loss: {loss.item():.4f}")
    epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch loss: {epoch_loss:.4f}")
    return epoch_loss

# Запуск
one_epoch()