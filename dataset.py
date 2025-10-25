import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import random
from image_folder import make_dataset

import random
random.seed(11)

class Medical_Dataset(data.Dataset):
    def __init__(self, A_path=None, B_path=None, phase=None, paired=True, transform=None):
        self.transform = transform
        assert A_path is not None and B_path is not None, 'two modalities should be used'

        self.dir_B = B_path
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.B_size = len(self.B_paths)

        self.dir_A = A_path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.A_size = len(self.A_paths)
        
        print(self.A_size, self.B_size)

        if not paired:
            random.shuffle(self.B_paths)

        self.paired = paired
        self.phase = phase

        assert self.A_size == self.B_size or not self.paired, 'paird but A_size != B_size'

    def __getitem__(self, i):
        A_path = self.A_paths[i % self.A_size].replace('\\', '/')
        A_image = Image.open(A_path).convert('RGB')
        B_path = self.B_paths[i % self.B_size].replace('\\', '/')
        B_image = Image.open(B_path).convert('RGB')

        if self.transform is not None:
            B_image = self.transform(B_image)
            A_image = self.transform(A_image)
            
        if self.phase == 'train':
            return A_image, B_image
        else:
            A_name = A_path.split('/')[-1].split('.')[0]
            B_name = B_path.split('/')[-1].split('.')[0]
            return A_name, B_name, A_image, B_image

    def __len__(self):
        return max(self.B_size, self.A_size)
    
def CreateDatasetSynthesis(phase, args):
    transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    dataset = Medical_Dataset(A_path=os.path.join(args.input_path, phase + 'A'), B_path=os.path.join(args.input_path, phase + 'B'), phase=phase, paired=False if phase == 'train' else True, transform=transform)
    return dataset