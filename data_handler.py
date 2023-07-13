import random

import numpy as np
from PIL import Image, ImageFile
import math

import os
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

script_dir = os.path.dirname(__file__)


def store_data(all_img, all_labels, file_names, nr_splits):
    unique_labels = list(set(all_labels))
    unique_labels_train_val = unique_labels[:math.ceil(len(unique_labels) * 0.5)]
    unique_labels_test = unique_labels[math.ceil(len(unique_labels) * 0.5):]
    x, y, filenames = [], [], []

    test_path = os.path.join(script_dir, "result", "test")
    train_path = os.path.join(script_dir, "result", "train_val")

    print("Generating Test Set")
    for index, img in enumerate(all_img):
        current_label = all_labels[index]
        if current_label in unique_labels_test:
            path = os.path.join(test_path, file_names[index])
            Image.fromarray(np.uint8(img)).save(path)
        elif current_label in unique_labels_train_val:
            x.append(img)
            y.append(current_label)
            filenames.append(file_names[index])

    random.seed(448)
    random.shuffle(unique_labels_train_val)
    for i in range(nr_splits):
        os.makedirs(os.path.join(train_path, str(i)))

    print("Generating Train/Valid Sets")
    label_splits = np.array_split(unique_labels_train_val, nr_splits)
    for idx, img in enumerate(x):
        current_label = y[idx]
        for split_idx, split in enumerate(label_splits):
            if current_label in split:
                path = os.path.join(train_path, str(split_idx), filenames[idx])
                Image.fromarray(np.uint8(img)).save(path)
                break


class CVLDataGenerator:
    labels = []

    def __init__(self, nr_splits):
        img_path = os.path.join(script_dir, "raw_data", "cvl")
        all_img = []
        all_labels = []
        file_names = []
        for img_name in tqdm(os.listdir(img_path), desc="Loading Data"):
            with Image.open(os.path.join(img_path, img_name)) as pil_img:
                label = int(img_name.split("-")[0])
                if label not in self.labels:
                    self.labels.append(label)
                label = self.labels.index(label)
                img = np.array(pil_img)

                all_img.append(img)
                all_labels.append(label)
                file_names.append(img_name)

        store_data(all_img, all_labels, file_names, nr_splits)


class FiremakerDataGenerator:
    labels = []

    def __init__(self, nr_splits):
        img_path = os.path.join(script_dir, "raw_data", "firemaker")
        all_img = []
        all_labels = []
        file_names = []
        for img_name in tqdm(os.listdir(img_path), desc="Loading Data"):
            with Image.open(os.path.join(img_path, img_name)) as pil_img:
                label = int(img_name[0:3])
                if label not in self.labels:
                    self.labels.append(label)
                label = self.labels.index(label)

                img = np.array(pil_img)

                all_img.append(img)
                all_labels.append(label)
                file_names.append(img_name)

        store_data(all_img, all_labels, file_names, nr_splits)


class Icdar2013DataGenerator:
    def __init__(self, nr_splits):
        img_path = os.path.join(script_dir, "raw_data", "icdar2013")
        all_img = []
        all_labels = []
        file_names = []
        for img_name in tqdm(os.listdir(img_path), desc="Loading Data"):
            with Image.open(os.path.join(img_path, img_name)) as pil_img:
                label = int(img_name.split("_")[0])

                img = np.array(pil_img)
                img = img *255

                all_img.append(img)
                all_labels.append(label)
                file_names.append(img_name)

        store_data(all_img, all_labels, file_names, nr_splits)
