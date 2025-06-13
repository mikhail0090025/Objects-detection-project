import kagglehub
import os
from lxml import objectify
import numpy as np

# Download latest version
path = kagglehub.dataset_download("akashs2021/general-object-detection")

annotations_folder = os.path.join(path, 'Scraped Images', 'Laptop', 'annotations')
images_folder = os.path.join(path, 'Scraped Images', 'Laptop', 'images')
input_images = []
output_laptops = []

for file in os.listdir(annotations_folder):
    file_path = os.path.join(annotations_folder, file)
    with open(file_path, "r") as xml:
        xml_read = xml.read()
        annotation = objectify.fromstring(xml_read)
        
        # Форматированный вывод
        print(f"\n--- Annotation for {file} ---")
        print(f"Filename: {annotation.filename.text}")
        print(f"Size: {annotation.size.width.text}x{annotation.size.height.text}, Depth: {annotation.size.depth.text}")
        for obj in annotation.object:
            name = obj.name.text
            xmin = obj.bndbox.xmin.text
            ymin = obj.bndbox.ymin.text
            xmax = obj.bndbox.xmax.text
            ymax = obj.bndbox.ymax.text

            norm_xmin = int(xmin) / annotation.size.width
            norm_ymin = int(ymin) / annotation.size.height
            norm_xmax = int(xmax) / annotation.size.width
            norm_ymax = int(ymax) / annotation.size.height

            output_laptops.append([norm_xmin, norm_ymin, norm_xmax, norm_ymax])

            print(f"Object: {name}, Box: (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}) \nnorm_xmin={norm_xmin}, norm_ymin={norm_ymin}, norm_xmax={norm_xmax}, norm_ymax={norm_ymax}")

for img in os.listdir(images_folder):
    file_path = os.path.join(images_folder, img)
    from PIL import Image
    image = Image.open(file_path).resize((200,200), Image.Resampling.LANCZOS).convert("RGB")
    img_array = (np.array(image) / 127.5).astype(np.float32)
    img_array = img_array - 1
    input_images.append(img_array)

input_images = np.array(input_images)
output_laptops = np.array(output_laptops)
print(input_images.shape)
print(output_laptops.shape)
print("Path to dataset files:", path)