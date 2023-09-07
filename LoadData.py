# Python 3.10.9
# 本程序中的函数"combine_img_and_label"将图像文件夹和对应的患者标签合并组成原始数据集
# STSDataset类用于生成用于模型训练的数据集
from PIL import Image
import pandas as pd
import os
# from torchvision import transforms
import torch
from collections import Counter


def combine_img_and_label(img_path, label_file):
    data_tuple = []
    label_pd = pd.read_excel(label_file).astype(str)
    label_map = {f: l for f, l in zip(label_pd['filename'], label_pd['Label'])}
    subdirs = os.listdir(img_path)
    for filename in subdirs:
        print(f"processing file directory {filename} ......")
        rela_path = os.path.join(img_path, filename)
        a, b = os.listdir(rela_path)
        img1, img2 = os.path.join(rela_path, a), os.path.join(rela_path, b)
        if 't2' in a:
            img1, img2 = img2, img1
        img1, img2 = Image.open(img1), Image.open(img2)
        data_tuple.append((img1, img2, label_map[filename], filename))
    return data_tuple


class STSDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, transform):
        self.raw_data = raw_data
        self.transform = transform
        self.image_t1s = torch.stack([self.preprocess(image) for image, _, _, _ in self.raw_data], dim=0)
        self.image_t2s = torch.stack([self.preprocess(image) for _, image, _, _ in self.raw_data], dim=0)
        self.labels = torch.tensor([int(label) for _, _, label, _ in self.raw_data])
        print(f"{len(self.labels)} data instances exists in this dataset.")
        print(f"The distribution of labels in dataset: {Counter(self.labels.tolist())}")

    def __len__(self):
        return len(self.image_t1s)

    def preprocess(self, image):
        return self.transform(image)

    def __getitem__(self, idx):
        return self.image_t1s[idx], self.image_t2s[idx], self.labels[idx]


class STSDatasetForT1(torch.utils.data.Dataset):
    def __init__(self, raw_data, transform):
        self.transform = transform
        self.image_t1s = torch.stack([self.preprocess(image) for image, _, _, _ in raw_data], dim=0)
        self.labels = torch.tensor([int(label) for _, _, label, _ in raw_data])
        print(f"{len(self.labels)} data instances exists in this dataset.")
        print(f"The distribution of labels in dataset: {Counter(self.labels.tolist())}")

    def __len__(self):
        return len(self.image_t1s)

    def preprocess(self, image):
        return self.transform(image)

    def __getitem__(self, idx):
        return self.image_t1s[idx], self.labels[idx]


class STSDatasetForT2(torch.utils.data.Dataset):
    def __init__(self, raw_data, transform):
        self.transform = transform
        self.image_t2s = torch.stack([self.preprocess(image) for _, image, _, _ in raw_data], dim=0)
        self.labels = torch.tensor([int(label) for _, _, label, _ in raw_data])
        print(f"{len(self.labels)} data instances exists in this dataset.")
        print(f"The distribution of labels in dataset: {Counter(self.labels.tolist())}")

    def __len__(self):
        return len(self.image_t2s)

    def preprocess(self, image):
        return self.transform(image)

    def __getitem__(self, idx):
        return self.image_t2s[idx], self.labels[idx]
