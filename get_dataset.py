import kagglehub
import os
from lxml import objectify
import numpy as np
from PIL import Image

# Download latest version
path = kagglehub.dataset_download("akashs2021/general-object-detection")

input_images = []  # Список пар (имя, массив)
outputs = []  # Список пар (изображение → список объектов)

all_folders = os.listdir(os.path.join(path, 'Scraped Images'))
all_folders = [os.path.join(path, 'Scraped Images', folder) for folder in all_folders]
categories_count = len(all_folders)
one_object_output_length = categories_count + 4

def empty_object_output():
    res = [0,0,0,0]
    for _ in range(categories_count):
        res.append(1.0 / categories_count)
    return res

def one_hot_category(category_id):
    return [1 if category_id == i else 0 for i in range(categories_count)]

def add_images(images_folder):
    for img in os.listdir(images_folder):
        file_path = os.path.join(images_folder, img)
        image = Image.open(file_path).resize((200, 200), Image.Resampling.LANCZOS).convert("RGB")
        img_array = (np.array(image) / 127.5).astype(np.float32) - 1
        input_images.append((img, img_array))  # Сохраняем имя и массив

print("Categories:", all_folders)
for category_id, folder in enumerate(all_folders):
    annotations_folder = os.path.join(folder, 'annotations')
    images_folder = os.path.join(folder, 'images')
    print(f"Processing folder: {os.listdir(folder)}")
    add_images(images_folder)
    
    # Словарь для соответствия изображений и аннотаций
    img_to_ann = {}
    for file in os.listdir(annotations_folder):
        file_path = os.path.join(annotations_folder, file)
        with open(file_path, "r") as xml:
            xml_read = xml.read()
            annotation = objectify.fromstring(xml_read)
            filename = annotation.filename.text  # Имя без пути
            objects = []
            
            print(f"\n--- Annotation for {file} ---")
            print(f"Filename: {filename}")
            print(f"Size: {annotation.size.width.text}x{annotation.size.height.text}, Depth: {annotation.size.depth.text}")
            
            for obj in annotation.object:
                name = obj.name.text
                xmin = int(obj.bndbox.xmin.text)
                ymin = int(obj.bndbox.ymin.text)
                xmax = int(obj.bndbox.xmax.text)
                ymax = int(obj.bndbox.ymax.text)
                width = int(annotation.size.width.text)
                height = int(annotation.size.height.text)

                norm_xmin = xmin / width
                norm_ymin = ymin / height
                norm_xmax = xmax / width
                norm_ymax = ymax / height

                output = [norm_xmin, norm_ymin, norm_xmax, norm_ymax]
                output.extend(one_hot_category(category_id))

                objects.append(output)
                print(f"Object: {name}, Box: (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})")
                print(f"Normalized: {output}")

            img_to_ann[filename] = objects
    
    # Соединяем аннотации с изображениями
    for img_name, img_array in input_images:  # Перебираем все изображения
        if img_name in img_to_ann:  # Проверяем, есть ли аннотация
            outputs.append((img_array, img_to_ann[img_name]))

# Преобразуем input_images в массив (только массивы, имена не нужны)
input_images = np.array([img for _, img in input_images])
max_objects_per_img = max([len(output[1]) for output in outputs])
output_shape = (max_objects_per_img, one_object_output_length)
for output in outputs:
    while len(output[1]) < max_objects_per_img:
        output[1].append(empty_object_output())

print(len(outputs))
outputs = [output[1] for output in outputs]

outputs = np.array(outputs)

print("Input images shape:", input_images.shape)
print("Outputs shape:", outputs.shape)
print("Number of output annotations:", len(outputs))
print("Example output:", [len(output) for output in outputs])
print("Example output:", [len(output[0]) for output in outputs])
print("Max found objects:", max_objects_per_img)
print("Path to dataset files:", path)