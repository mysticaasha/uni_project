import os
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image


DATASET_PATH = 'data/original_data'
OUTPUT_DIR = 'data/pretrained_data'

TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

class_mapping = {
    'AnnualCrop': 0,
    'Forest': 1,
    'HerbaceousVegetation': 2,
    'Highway': 3,
    'Industrial': 4,
    'Pasture': 5,
    'PermanentCrop': 6,
    'Residential': 7,
    'River': 8,
    'SeaLake': 9
}

def split_data(class_images):
    train_images, test_images = train_test_split(class_images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)
    return train_images, val_images, test_images

def apply_pca(image_path, pca_model=None):
    with rasterio.open(image_path) as src:
        img = src.read()
    
    reshaped_img = img.reshape(13, -1)

    if pca_model is None:
        pca = PCA(n_components=3)
        img_pca = pca.fit_transform(reshaped_img.T).T
        return img_pca.reshape(3, 64, 64), pca
    else:
        img_pca = pca_model.transform(reshaped_img.T).T 
        return img_pca.reshape(3, 64, 64)

def process_split(split, images, class_name, pca_model=None):
    split_dir = {
        'train': TRAIN_DIR,
        'val': VAL_DIR,
        'test': TEST_DIR
    }[split]
    
    class_split_dir = os.path.join(split_dir, class_name)
    os.makedirs(class_split_dir, exist_ok=True)

    for img_path in tqdm(images, desc=f"Processing {class_name} - {split}"):
        if split == 'train':
            img_pca, pca_model = apply_pca(img_path)
        else:
            img_pca = apply_pca(img_path, pca_model)

        
        img_pca = (img_pca - img_pca.min()) / (img_pca.max() - img_pca.min())
        img_pca = (img_pca * 255).astype(np.uint8)

        image_name = os.path.basename(img_path).replace('.tif', '.jpg')
        image_data = Image.fromarray(img_pca.T, 'RGB')
        image_data.save(os.path.join(class_split_dir, image_name), format='JPEG')


    return pca_model

def prepare_dataset():
    pca_model = None
    for class_name in class_mapping.keys():
        class_dir = os.path.join(DATASET_PATH, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith('.tif')]
        image_paths = [os.path.join(class_dir, img) for img in images]

        train_images, val_images, test_images = split_data(image_paths)

        pca_model = process_split('train', train_images, class_name, pca_model)
        process_split('val', val_images, class_name, pca_model)
        process_split('test', test_images, class_name, pca_model)

prepare_dataset()
print("data preparation completed!")

from ultralytics import YOLO

DATASET_DIR = 'data/pretrained_data/'

model = YOLO('models/yolov8n-cls.pt')

model.train(
    data=DATASET_DIR,
    imgsz=64,
    task='classify',
    epochs=100,
    batch=16,
    save=True,
    device='cpu',
)

trained_model = YOLO('runs/classify/train4/weights/best.pt')
metrics = trained_model.val()
