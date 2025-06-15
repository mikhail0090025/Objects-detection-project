import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from get_dataset import input_images, spots_of_interest, max_objects_per_img, categories_count
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class SpotsOfInterestDataset(Dataset):
    def __init__(self, images, annotations):
        # Преобразуем изображения
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        # Преобразуем аннотации в фиксированный тензор
        self.annotations = torch.zeros((len(annotations), max_objects_per_img, 4), dtype=torch.float32)
        for i, anns in enumerate(annotations):
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
            nn.Conv2d(3, 16, 3, padding=1, stride=2),  # 200x200 => 100x100
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, padding=1, stride=1),  # 100x100 => 100x100
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1, stride=2),  # 100x100 => 50x50
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1, stride=1),  # 50x50 => 50x50
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),  # 50x50 => 25x25
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),  # 25x25 => 25x25
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),  # 25x25 => 25x25
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(25 * 25 * 32, max_objects_per_img * 4),
            nn.Linear(max_objects_per_img * 4, max_objects_per_img * 4),
            nn.Linear(max_objects_per_img * 4, max_objects_per_img * 4),
            nn.Unflatten(1, (max_objects_per_img, 4)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.clamp(0, 1)
        return x

class FullDetector(nn.Module):
    def __init__(self):
        super(FullDetector, self).__init__()
        self.objects_detector = ObjectsDetector()

    def forward(self, x):
        return self.objects_detector(x)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        rmse = torch.sqrt(l1 + 1e-7)
        return rmse

# Инициализация
train_images, val_images, train_outputs, val_outputs = train_test_split(
    input_images, spots_of_interest, test_size=0.2, random_state=42
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = SpotsOfInterestDataset(train_images, train_outputs)
val_dataset = SpotsOfInterestDataset(val_images, val_outputs)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
detector = FullDetector().to(device)
optim_detector = optim.Adam(detector.parameters(), lr=0.00002, weight_decay=0.01)
all_losses = []
all_val_losses = []

def rmse_loss(pred, target):
    return torch.sqrt(torch.nn.L1Loss(pred, target) + 1e-7)

# Использование
loss_fn = RMSELoss()
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