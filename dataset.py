import torch, os, torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class my_dataset(Dataset):
    def __init__(self, ds_dir, txt_name):
        self.ds_dir = ds_dir
        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.init_ImagesLabels()

    def init_ImagesLabels(self):
        images, labels = [], []

        txt_path = os.path.join(self.ds_dir, 'dataset_txt', self.txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        for line in data:
            line = line.replace('\\', os.sep)
            line = line.strip()
            contents = line.split()

            image_path = os.path.join(self.ds_dir, contents[0])
            images.append(image_path)
            labels.append(contents[-1])

        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        image = self.img_transforms(image)
        label = np.array(label).astype(np.int64)

        return image, label


if __name__ == '__main__':
    # ds_dir = r'/veracruz/home/j/jwang/data/Stage4_D2_CityPersons_7Augs'
    # txt_name = 'test.txt'
    # my_dataset = my_dataset(ds_dir, txt_name)
    # my_loader = DataLoader(my_dataset, batch_size=4)
    #
    # for images, labels in my_loader:
    #     print(labels)
    #     break

    print('torch:', torch.__version__)
    print('torchvision:', torchvision.__version__)
























