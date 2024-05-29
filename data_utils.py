from typing import Any
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

from torchvision.transforms import functional as F

import pandas as pd

import numpy as np
import random

def resample(_dataset, ratio = 3):
    min_size = _dataset['label'].value_counts().min()
    lst = []
    added_unique_rows = 0
    all_n_rows = 0

    for class_index, group in _dataset.groupby('label'):
        all_n_rows += len(group)
        if class_index == 0:
            added_unique_rows += min_size*ratio
            lst.append(group.sample(min_size*ratio, replace=False))
        else:
            if len(group) > min_size*ratio:
                added_unique_rows += min_size*ratio
                lst.append(group.sample(min_size*ratio, replace=False))
            else:
                lst.append(group)
                added_unique_rows += len(group)
                lst.append(group.sample(min_size*ratio-len(group), replace=True))

    _dataset = pd.concat(lst)

    for class_index, group in _dataset.groupby('label'):
        print(f'{class_index}: length: {len(group)}')

    print('N_added_rows: ', added_unique_rows)
    print('N_all_rows: ', all_n_rows)
    print('Ratio of used rows: ', added_unique_rows/all_n_rows)
    return _dataset


class Spot(object):
    def __init__(self, size, prob = 0.5):
        self.size = size
        self.prob = prob
        self.center = None
        self.radius = None
        self.zeros = torch.zeros((self.size, self.size)) #.cuda()
        self.ones = torch.ones((3, 1)) #.cuda()
        self.tensor_to_image = T.ToPILImage()

    def __call__(self, image_tensors, target = None):
        if random.random() < self.prob:

            modified_image_tensors = image_tensors.clone()
            n_spots = random.randint(5, 7)
            self.initial_mask = self.zeros.clone()

            self.dim1_offset = (image_tensors.shape[1] - self.size) // 2
            self.dim2_offset = (image_tensors.shape[2] - self.size) // 2

            for _ in range(n_spots):
                # image_tensors = self.add_random_spot(image_tensors)
                modified_image_tensors = self.add_random_spot(modified_image_tensors)
            # rerun self.tensor_to_image(torch.clamp(image_tensors, max = 255))
            return torch.clamp(modified_image_tensors, max = image_tensors.max())
        else: return image_tensors
        
    def add_random_spot(self, image_tensor):
        self.radius = random.randint(int(0.01 * self.size) + 1, int(0.05 * self.size))
        self.center = [random.randint(self.radius + 1, self.size - self.radius - 1), 
                       random.randint(self.radius + 1, self.size - self.radius - 1)]
        y, x = np.ogrid[: self.size, : self.size]
        dist_from_center = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        circle = dist_from_center <= (self.radius // 2)

        k = 14 / 25 + (1.0 - self.radius / 25)
        beta = 0.5 + (1.5 - 0.5) * self.radius / 25
        A = k * self.ones.clone()
        d = 0.3 * self.radius / 25
        t = np.exp(-beta * d)

        spot_mask = self.zeros.clone()
        spot_mask[circle] = torch.multiply(A[0], torch.tensor(1 - t))

        self.initial_mask = self.initial_mask + spot_mask
        self.initial_mask[self.initial_mask != 0] = 1

        sigma = (5 + (2 - 0) * self.radius / 25) * 2
        rad_w = random.randint(int(sigma / 5), int(sigma / 4))
        rad_h = random.randint(int(sigma / 5), int(sigma / 4))

        if (rad_w % 2) == 0: rad_w = rad_w + 1
        if (rad_h % 2) == 0: rad_h = rad_h + 1

        spot_mask = F.gaussian_blur(torch.reshape(spot_mask, (1, self.size, self.size)), (rad_w, rad_h), sigma)
        spot_mask = torch.stack([spot_mask, spot_mask, spot_mask]) * 10
        
        image_tensor[:, self.dim1_offset : self.dim1_offset + self.size, self.dim2_offset : self.dim2_offset + self.size] += torch.reshape(spot_mask, (3, self.size, self.size))
        return image_tensor




class Halo(object):
    def __init__(self, size, prob = 0.5, intensity_range = (0.8, 1.2)):
        self.size = size
        self.prob = prob
        self.center = None
        self.radius = None
        self.intensity_range = intensity_range
        self.tensor_to_image = T.ToPILImage()

    def __call__(self, image_tensors, target = None):
        # max_val = image_tensors
        if random.random() < self.prob:
            # print('Yes')
            modified_image_tensors = image_tensors.clone()
            # print(f'Min value: {torch.amin(modified_image_tensors)}')
            # print(f'Max value: {torch.amax(modified_image_tensors)}')
            n_halos = random.randint(5, 7)

            self.dim1_offset = (image_tensors.shape[1] - self.size) // 2
            self.dim2_offset = (image_tensors.shape[2] - self.size) // 2
            
            for _ in range(n_halos):
                modified_image_tensors = self.add_random_halo(modified_image_tensors)
                # modified_image_tensors = self.add_random_halo(modified_image_tensors)
            # return torch.clamp(modified_image_tensors, max = image_tensors.max())
            return modified_image_tensors
        # torch.clamp(modified_image_tensors, max = image_tensors.max())
            # return torch.clamp(modified_image_tensors, min = torch.amin(image_tensors), max = torch.amax(image_tensors))
        else: return image_tensors

    def add_random_halo(self, image_tensor):
        self.radius = random.randint(int(0.01 * self.size), int(0.05 * self.size))
        self.center = [random.randint(self.radius + 1, self.size - self.radius - 1),
                        random.randint(self.radius + 1, self.size - self.radius - 1)]
        
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        dist_from_center = torch.sqrt(((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        normalized_dist = dist_from_center / self.radius
        
        halo_intensity = torch.clamp(self.intensity_range[0] + (self.intensity_range[1] - self.intensity_range[0]) * (1 - normalized_dist), min = 0, max = 1)
        halo_mask = dist_from_center <= self.radius // 2
        halo_effect = halo_intensity * (self.radius - dist_from_center) / self.radius
        halo_effect = np.clip(halo_effect, 0, 1)
        halo_effect = np.expand_dims(halo_effect, axis = 0)
        halo_effect = np.repeat(halo_effect, image_tensor.shape[0], axis = 0)
        image_tensor[:, halo_mask] = image_tensor[:, halo_mask] * (1 - halo_effect[:, halo_mask]) + halo_effect[:, halo_mask]

        return image_tensor
    


class Hole(object):
    def __init__(self, size, prob = 0.5):
        self.size = size
        self.prob = prob
        self.center = None
        self.radius = None
        self.tensor_to_image = T.ToPILImage()

    def __call__(self, image_tensors, target = None):
        if random.random() < self.prob:
            # print('Yes')
            modified_image_tensors = image_tensors.clone()
            # print(f'Min value: {torch.amin(modified_image_tensors)}')
            # print(f'Max value: {torch.amax(modified_image_tensors)}')
            n_halos = random.randint(5, 7)

            self.dim1_offset = (image_tensors.shape[1] - self.size) // 2
            self.dim2_offset = (image_tensors.shape[2] - self.size) // 2
            
            for _ in range(n_halos):
                image_tensors = self.add_random_hole(image_tensors)
                # modified_image_tensors = self.add_random_hole(modified_image_tensors)
            return torch.clamp(image_tensors, min = torch.amin(image_tensors), max = torch.amax(image_tensors))
            # return torch.clamp(modified_image_tensors, min = torch.amin(image_tensors), max = torch.amax(image_tensors))
        else: return image_tensors

    def add_random_hole(self, image_tensor):
        self.radius = random.randint(int(0.01 * self.size), int(0.05 * self.size))
        self.center = [random.randint(self.radius + 1, self.size - self.radius - 1),
                        random.randint(self.radius + 1, self.size - self.radius - 1)]
        
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        dist_from_center = torch.sqrt(((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        
        hole_mask = dist_from_center <= self.radius // 2
        image_tensor[:, hole_mask] = 0

        return image_tensor

class Blur(object):
    def __init__(self, sigma=None, prob = 0.5):
        self.level_range = [0.1, 1]
        self.sigma = sigma
        self.prob = prob

    def __get_parameter(self):
        return np.random.uniform(self.level_range[0], self.level_range[1])

    def __call__(self, image, target=None):
        if random.random() < self.prob:
           
            rad_w = random.randint(10, 50)
            if (rad_w % 2) == 0: rad_w = rad_w + 1
            rad_h = rad_w
            image = F.gaussian_blur(image, (rad_w,rad_h))

        return image
    
from PIL import ImageEnhance

def RandomSharpen(image, alpha = 0.2):
    sharpener = ImageEnhance.Sharpness(image)
    factor = 0.5  
    image = sharpener.enhance(1.0 + alpha * factor)
    return image


class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            # Size must match the ratio while cropping to the edge of the image
            # print(ratio, w, h)
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w>=h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return T.functional.center_crop(img, size)
