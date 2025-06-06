import os
import pandas as pd
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, root=None, split='labelled_train', num=None, transform=None):
        self.root = os.path.join("/content/drive/MyDrive/newdataset/csv_data")
        self.list = os.path.join(self.root, "trainset.csv")
        self.split = split
        self.transform = transform
        self.sample_list = []

        if self.split == "labelled_train":
            csv = pd.read_csv(self.list)
            for i in range(num):
                filename = csv.iloc[i, 0]
                self.sample_list.append(filename)

        if self.split == "unlabelled_train":
            csv = pd.read_csv(self.list)
            for i in range(num):    
                filename = csv.iloc[i, 1]
                self.sample_list.append(filename)
                
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_name = self.sample_list[idx]
        path = os.path.join("/content/drive/MyDrive/newdataset/Dataset/Training Set/{}/mri_norm2.h5".format(image_name))
        h5f = h5py.File(path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] < self.output_size[0] or label.shape[1] < self.output_size[1] or label.shape[2] < self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        depth, height, width = image.shape
        target_d, target_h, target_w = self.output_size

        d_start = random.randint(0, depth - target_d)
        h_start = random.randint(0, height - target_h)
        w_start = random.randint(0, width - target_w)

        image = image[d_start:d_start+target_d, h_start:h_start+target_h, w_start:w_start+target_w]
        label = label[d_start:d_start+target_d, h_start:h_start+target_h, w_start:w_start+target_w]

        return {'image': image, 'label': label}


class RandomRotFlip:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        axis = random.randint(0, 2)
        k = random.randint(0, 3)
        image = np.rot90(image, k, axes=(axis, (axis+1)%3)).copy()
        label = np.rot90(label, k, axes=(axis, (axis+1)%3)).copy()
        if random.random() > 0.5:
            flip_axis = random.randint(0, 2)
            image = np.flip(image, axis=flip_axis).copy()
            label = np.flip(label, axis=flip_axis).copy()
        return {'image': image, 'label': label}




class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class ToTensor:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.copy()).float().unsqueeze(0)
        label = torch.from_numpy(label.copy()).long()
        return {'image': image, 'label': label}



class CreateOnehotLabel(object):
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self, predictions):
        label = predictions.detach().cpu().numpy()
        onehot_label = np.zeros((1, self.num_classes, label.shape[2], label.shape[3], label.shape[4]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[0, i, :, :, :] = (label[0][i] == i).astype(np.float32)
        return torch.from_numpy(onehot_label)


def main():
    # Initialize dataset
    dataset = LAHeart(root="/content/drive/MyDrive/newdataset/csv_data", split="labelled_train", num=10)

    # Print dataset size (number of samples)
    print(f"Dataset size: {len(dataset)}")

    # Get a sample item and print its shape
    sample = dataset[0]
    print("Sample image shape:", sample['image'].shape)
    print("Sample label shape:", sample['label'].shape)

if __name__ == "__main__":
    main()
