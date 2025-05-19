import json
from datasets import Dataset
from PIL import Image
import os

JSON_FILE = "./TextAtlas5M/InternData/data_info.json"

def save_data(data):
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def add_objects(data):   
    d = []
    for row in data:
        new_object = {
            "height": 512,
            "width": 512,
            "ratio": 1.0,
            "path": row['image_path'],
            "prompt": row['annotation'],
            "sharegpt4v": ""
        }
        d.append(new_object)
        row['image'].save(f'./TextAtlas5M/InternImgs/{row["image_path"]}', 'JPEG')

    return d

root_dir = './TextAtlas5m'
info_dir = './TextAtlas5m/InternData'
data_dir = './TextAtlas5m/InternImgs'

os.makedirs(root_dir, exist_ok=True)
os.makedirs(info_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

for i in range(86):
    s = str(i)
    data_to_json = []

    if i < 10:
        s = f"0{s}"

    ds = Dataset.from_file(f"./huggingface/CSU-JPG___text_atlas5_m/TextVisionBlend/0.0.0/945d5e3053619ea80eb8e616d6cdddd85e7aa4ef/text_atlas5_m-train-000{s}-of-00086.arrow")

    data_to_json.extend(add_objects(ds))

    if i == 2:
        break

save_data(data_to_json)