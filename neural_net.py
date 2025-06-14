import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomObjectDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.image_files.sort()  # Для согласованности

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + '.txt')

        # Загружаем изображение
        image = Image.open(img_path).convert("RGB")
        
        # Читаем аннотации
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([class_id, x_center, y_center, width, height])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, boxes

# Пример трансформации
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Уже ресайзим в 200x200
    transforms.ToTensor(),          # Конвертируем в тензор [0, 1]
])

# Создание датасета и загрузчика
dataset = CustomObjectDataset(images_dir="dataset/images", labels_dir="dataset/labels", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Пример использования
for images, targets in dataloader:
    print("Images shape:", images.shape)  # [batch_size, 3, 200, 200]
    print("Targets shape:", targets.shape)  # [batch_size, num_objects, 5]
    break