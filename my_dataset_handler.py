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
            obj_class = annotation['class']
            obj_box = annotation['box']
            obj_box[0] = obj_box[0] / initial_size[0]
            obj_box[2] = obj_box[2] / initial_size[0]
            obj_box[1] = obj_box[1] / initial_size[1]
            obj_box[3] = obj_box[3] / initial_size[1]
            self.output.append(obj_box)
        while len(self.output) < max_objects:
            self.output.append([0,0,0,0])
        
        self.output = np.array(self.output)
    
    def flip(self, x: bool, y: bool):
        if x:
            self.image = [np.flip(row) for row in self.image]
            for box in self.output:
                box[0] = 1 - box[0]
                box[2] = 1 - box[2]
                box[0], box[2] = box[2], box[0]
        if y:
            self.image = np.flip(self.image)
            for box in self.output:
                box[1] = 1 - box[1]
                box[3] = 1 - box[3]
                box[1], box[3] = box[1], box[3]


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

def add_all_flipped_variants():
    global all_images
    if all_images is not None and len(all_images) > 0:
        x_flipped = np.copy(all_images)
        y_flipped = np.copy(all_images)
        xy_flipped = np.copy(all_images)
        for img in x_flipped:
            img.flip(True, False)
        for img in y_flipped:
            img.flip(False, True)
        for img in xy_flipped:
            img.flip(True, True)
        all_images = np.append(all_images, x_flipped)
        all_images = np.append(all_images, y_flipped)
        all_images = np.append(all_images, xy_flipped)
        print(f"all_images: {len(all_images)}")
    else:
        raise ValueError("all_images is None or empty")

if os.path.exists("dataset.npz"):
    load_dataset()
else:
    build_dataset()
    add_all_flipped_variants()
    save_dataset()

print("Dataset is ready")