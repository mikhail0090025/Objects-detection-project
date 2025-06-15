import numpy as np
import os
import json

path_to_annotations = os.path.join('MyDataset', 'annotations.json')
with open(path_to_annotations, "r") as file:
    annotations = json.load(file)
    print(annotations['images'])
    print(type(annotations['images']))