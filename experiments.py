import torch
import math
'''
# Пример матрицы
A = torch.tensor([
    [i for i in range(0, 4)],
    [i for i in range(4, 8)],
    [i for i in range(8, 12)],
    [i for i in range(12, 16)],
], dtype=torch.float32)

# SVD
U, S, V = torch.svd(A)

print("A:\n", A)
print("U:\n", U)
print("S:\n", S)  # Сингулярные значения
print("V:\n", V)
'''

def generate_anchors(feature_map_size, img_size, scales=[128, 256, 512], ratios=[0.5, 1, 2]):
    # feature_map_size: [H, W] размера feature map
    # img_size: [H, W] исходного изображения
    # scales и ratios для создания якорей
    anchors = []
    for h in range(feature_map_size[0]):
        for w in range(feature_map_size[1]):
            for scale in scales:
                for ratio in ratios:
                    # Центр якоря в пространстве feature map
                    center_h = h + 0.5
                    center_w = w + 0.5
                    
                    # Переводим в пиксели изображения (учитываем stride, например, 16)
                    stride = img_size[0] / feature_map_size[0]
                    center_h_px = center_h * stride
                    center_w_px = center_w * stride
                    
                    # Вычисляем размеры якоря
                    h_ = math.sqrt(scale * scale / ratio)
                    w_ = h_ * ratio
                    
                    # Координаты [x1, y1, x2, y2]
                    x1 = center_w_px - w_ / 2
                    y1 = center_h_px - h_ / 2
                    x2 = center_w_px + w_ / 2
                    y2 = center_h_px + h_ / 2
                    
                    anchors.append([x1, y1, x2, y2])
    
    return torch.tensor(anchors, dtype=torch.float32)

anchors = generate_anchors([7,7], [1000,1000])
print(anchors)
print('anchors.shape')
print(anchors.shape)