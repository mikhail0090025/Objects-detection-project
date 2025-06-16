import numpy as np
import os
import json
from PIL import Image

path_to_annotations = os.path.join('MyDataset', 'annotations.json')
annotations = None
initial_size = (1008 * 4, 477 * 4)
start_size = (1008, 477)
max_objects = 7
all_images = []

class OneObject:
    def __init__(self, image_path, annotations):
        img = Image.open(image_path).resize(start_size, Image.Resampling.LANCZOS).convert("RGB")
        self.image = (np.array(img) / 127.5).astype(np.float32) - 1
        self.output = []
        for annotation in annotations:
            obj_box = annotation['box']
            obj_class = annotation['class']
            if max(obj_box) > 1:
                obj_box[0] = obj_box[0] / initial_size[0]
                obj_box[2] = obj_box[2] / initial_size[0]
                obj_box[1] = obj_box[1] / initial_size[1]
                obj_box[3] = obj_box[3] / initial_size[1]
            self.output.append(obj_box)
        while len(self.output) < max_objects:
            self.output.append([0,0,0,0])
        print(self.output)
        
        self.output = np.array(self.output)
    
    def flip(self, x: bool = False, y: bool = False):
        # Создаём копию изображения для избежания побочных эффектов
        flipped_image = np.copy(self.image)
        flipped_output = np.copy(self.output)
        
        # Горизонтальный переворот
        if x:
            flipped_image = np.flip(flipped_image, axis=1)  # По ширине
            for box in flipped_output:
                if box.sum() > 0:
                    box[0], box[2] = 1 - box[2], 1 - box[0]  # Инверсия xmin и xmax

        # Вертикальный переворот
        if y:
            flipped_image = np.flip(flipped_image, axis=0)  # По высоте
            for box in flipped_output:
                if box.sum() > 0:
                    box[1], box[3] = 1 - box[3], 1 - box[1]  # Инверсия ymin и ymax

        # Обновляем только если был флип
        if x or y:
            self.image = flipped_image
            self.output = flipped_output
            print(self.output)

def build_dataset():
    global all_images
    with open(path_to_annotations, "r") as file:
        annotations = json.load(file)['images']

    for image in annotations:
        image_name = image['file']
        image_annotations = image['annotations']
        image_path = os.path.join('MyDataset', image_name)
        print(f"Image name: {image_name}")
        all_images.append(OneObject(image_path, image_annotations))
        all_images.append(OneObject(image_path, image_annotations))
        all_images.append(OneObject(image_path, image_annotations))
        all_images.append(OneObject(image_path, image_annotations))
        all_images[-1].flip(True, True)
        all_images[-2].flip(False, True)
        all_images[-3].flip(True, False)

    all_images = np.array(all_images)

def save_dataset():
    if all_images is not None and len(all_images) > 0:
        np.savez_compressed("dataset.npz", all_images=all_images)
    else:
        raise ValueError("all_images is None or empty, it cant be saved")

def load_dataset():
    global all_images
    if os.path.exists("dataset.npz"):
        all_images = np.load("dataset.npz", allow_pickle=True)['all_images']
    else:
        raise ValueError("File was not found")

if os.path.exists("dataset.npz"):
    load_dataset()
else:
    build_dataset()
    save_dataset()

print("Dataset is ready")