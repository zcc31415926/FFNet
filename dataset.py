from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os


def isValidBBox(l, t, r, b):
    if l < 0 or t < 0 or r < 0 or b < 0:
        return False
    elif l >= r or t >= b:
        return False
    else:
        return True


def transform(img_size):
    return transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class KITTIDataset(Dataset):
    def __init__(self, root, img_size=224, train=True):
        image_dir = os.path.join(root, 'kitti_data/training/image_2')
        label_dir = os.path.join(root, 'kitti_data/training/label_2')
        num_data = len(os.listdir(label_dir))
        num_train = num_data * 4 // 5
        if train:
            images = [os.path.join(image_dir, f'{str(i).zfill(6)}.png') for i in range(num_train)]
            labels = [os.path.join(label_dir, f'{str(i).zfill(6)}.txt') for i in range(num_train)]
        else:
            images = [os.path.join(image_dir, f'{str(i).zfill(6)}.png') for i in range(num_train, num_data)]
            labels = [os.path.join(label_dir, f'{str(i).zfill(6)}.txt') for i in range(num_train, num_data)]
        self.img_size = img_size
        self.images = []
        self.bbox_2d = []
        self.bbox_3d = []
        self.orientations = []
        for i in range(len(images)):
            label_file = labels[i]
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    contents = line.strip().split(' ')
                    if contents[0] == 'Pedestrian':
                        left, top, right, bottom = np.array(contents[4 : 8]).astype(np.float64).astype(np.int)
                        if isValidBBox(left, top, right, bottom):
                            self.images.append(images[i])
                            self.bbox_2d.append([left, top, right, bottom])
                            height, width, length = np.array(contents[8 : 11]).astype(np.float64)
                            self.bbox_3d.append([height, width, length])
                            self.orientations.append(float(contents[3]))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        w, h = img.size
        l, t, r, b = self.bbox_2d[idx]
        l = max(l, 0)
        t = max(t, 0)
        r = min(r, w)
        b = min(b, h)
        long_edge = max(b - t, r - l)
        img = np.array(img)[t : b, l : r]
        img = transform(self.img_size)(Image.fromarray(img))
        bbox_2d = torch.Tensor([b - t, r - l]).float() / long_edge
        h, w, l = self.bbox_3d[idx]
        long_edge = max(h, w, l)
        bbox_3d = torch.Tensor(self.bbox_3d[idx]).float() / long_edge
        orientation = torch.Tensor([self.orientations[idx]]).float()
        return img, bbox_2d, bbox_3d, orientation

    def __len__(self):
        return len(self.images)


class KITTITest(Dataset):
    def __init__(self, root, results_2d_dir, img_size=224):
        image_dir = os.path.join(root, 'kitti_data/testing/image_2')
        num_test = len(os.listdir(image_dir))
        self.img_size = img_size
        self.images = [os.path.join(image_dir, f'{str(i).zfill(6)}.png') for i in range(num_test)]
        self.results_2d = [os.path.join(results_2d_dir, f'{str(i).zfill(6)}.txt') for i in range(num_test)]

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        w, h = img.size
        img = np.array(img)
        results_2d_file = self.results_2d[idx]
        imgs = []
        bbox_2d = []
        results_2d_row = []
        with open(results_2d_file, 'r') as f:
            for j, line in enumerate(f.readlines()):
                contents = line.strip().split(' ')
                if contents[0] == 'Pedestrian':
                    l, t, r, b = np.array(contents[4 : 8]).astype(np.float64).astype(np.int)
                    if isValidBBox(l, t, r, b):
                        l = max(l, 0)
                        t = max(t, 0)
                        r = min(r, w)
                        b = min(b, h)
                        patch = img[t : b, l : r]
                        patch = transform(self.img_size)(Image.fromarray(patch))
                        imgs.append(patch.unsqueeze(0))
                        long_edge = max(b - t, r - l)
                        bbox_2d.append(torch.Tensor([b - t, r - l]).float().unsqueeze(0) / long_edge)
                        results_2d_row.append(j)
        if len(imgs) == 0:
            return None, None, results_2d_file, None
        else:
            return torch.cat(imgs, dim=0), torch.cat(bbox_2d, dim=0), results_2d_file, results_2d_row

    def __len__(self):
        return len(self.images)

