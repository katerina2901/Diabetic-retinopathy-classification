from typing import Any
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

from torchvision.transforms import functional as F

import pandas as pd

import numpy as np
import random

# Resample dataset to balance class distribution
def resample(_dataset, ratio = 3):
    # Calculate the size of minority class
    min_size = _dataset['label'].value_counts().min()
    lst = []
    added_unique_rows = 0
    all_n_rows = 0

    # For each class, oversample/undersample to min_size * ratio
    for class_index, group in _dataset.groupby('label'):
        all_n_rows += len(group)
        if class_index == 0:
            # If class index is 0, sample with the specified ratio without replacement
            added_unique_rows += min_size*ratio
            lst.append(group.sample(min_size*ratio, replace=False))
        else:
            if len(group) > min_size*ratio:
                # If group size is larger than the desired size, sample without replacement
                added_unique_rows += min_size*ratio
                lst.append(group.sample(min_size*ratio, replace=False))
            else:
                # otherwise, sample with the replacement to reach the desired size
                lst.append(group)
                added_unique_rows += len(group)
                lst.append(group.sample(min_size*ratio-len(group), replace=True))

    # Concatenate the sampled subsets
    _dataset = pd.concat(lst)

    # Calculate and display the size of resampled classes
    for class_index, group in _dataset.groupby('label'):
        print(f'{class_index}: length: {len(group)}')

    # Display the count of instances per class and the ratio of added to the total rows
    print('N_added_rows: ', added_unique_rows)
    print('N_all_rows: ', all_n_rows)
    print('Ratio of used rows: ', added_unique_rows/all_n_rows)

    # Return the balanced dataset
    return _dataset


### Post Transform Augmentation

# Augment images by randomly adding spots
class Spot(object):
    def __init__(self, size, prob = 0.5):
        self.size = size
        self.prob = prob
        self.center = None
        self.radius = None
        self.zeros = torch.zeros((self.size, self.size)) # Create a zero tensor for the mask
        self.ones = torch.ones((3, 1)) # Create a tensor with ones for spot initially
        self.tensor_to_image = T.ToPILImage()

    def __call__(self, image_tensors, target = None):
        # Randomly decide whether to add spots based on a probability
        if random.random() < self.prob:

            # Use modified_image_tensors for visualization purpose
            modified_image_tensors = image_tensors.clone()

            # Generate random number of spots between 5 and 7
            n_spots = random.randint(5, 7)

            # initialize the mask
            self.initial_mask = self.zeros.clone()

            self.dim1_offset = (image_tensors.shape[1] - self.size) // 2
            self.dim2_offset = (image_tensors.shape[2] - self.size) // 2

            # Add random spots on the image tensors
            for _ in range(n_spots):
                # image_tensors = self.add_random_spot(image_tensors)
                modified_image_tensors = self.add_random_spot(modified_image_tensors)
            # rerun self.tensor_to_image(torch.clamp(image_tensors, max = 255))
            return torch.clamp(modified_image_tensors, max = image_tensors.max()) # Clamp the values to maintain a valid range
        else: return image_tensors

    # Function for adding random spot
    def add_random_spot(self, image_tensor):
        # Randomly generate radius and center points for spot
        self.radius = random.randint(int(0.01 * self.size) + 1, int(0.05 * self.size))
        self.center = [random.randint(self.radius + 1, self.size - self.radius - 1), 
                       random.randint(self.radius + 1, self.size - self.radius - 1)]
        y, x = np.ogrid[: self.size, : self.size]
        dist_from_center = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)

        # Create a circular mask
        circle = dist_from_center <= (self.radius // 2)

        k = 14 / 25 + (1.0 - self.radius / 25) # Calculate intensity factor
        beta = 0.5 + (1.5 - 0.5) * self.radius / 25 # Calculate intensity decay factor
        A = k * self.ones.clone() # Initialize the intensity tensor
        d = 0.3 * self.radius / 25
        t = np.exp(-beta * d) # Calculate exponential decay

        # Apply the spot effect to the mask
        spot_mask = self.zeros.clone()
        spot_mask[circle] = torch.multiply(A[0], torch.tensor(1 - t))

        # Update the initial mask with spot
        self.initial_mask = self.initial_mask + spot_mask
        self.initial_mask[self.initial_mask != 0] = 1

        # Calculate sigma for blurring and kernel size
        sigma = (5 + (2 - 0) * self.radius / 25) * 2
        rad_w = random.randint(int(sigma / 5), int(sigma / 4))
        rad_h = random.randint(int(sigma / 5), int(sigma / 4))

        if (rad_w % 2) == 0: rad_w = rad_w + 1
        if (rad_h % 2) == 0: rad_h = rad_h + 1

        # Apply Gaussian blur to create the spot effect
        spot_mask = F.gaussian_blur(torch.reshape(spot_mask, (1, self.size, self.size)), (rad_w, rad_h), sigma)
        spot_mask = torch.stack([spot_mask, spot_mask, spot_mask]) * 10

        # Add spot to the original image tensor
        image_tensor[:, self.dim1_offset : self.dim1_offset + self.size, self.dim2_offset : self.dim2_offset + self.size] += torch.reshape(spot_mask, (3, self.size, self.size))

        # Return the augmented image tensor
        return image_tensor 



# Augment images by randomly adding halo effects
class Halo(object):
    def __init__(self, size, prob = 0.5, intensity_range = (0.8, 1.2)):
        self.size = size
        self.prob = prob
        self.center = None
        self.radius = None
        self.intensity_range = intensity_range
        self.tensor_to_image = T.ToPILImage()

    def __call__(self, image_tensors, target = None):
        # Randomly decide whether to add halos based on a probability
        if random.random() < self.prob:
            # Use modified_image_tensors for visualization purpose
            modified_image_tensors = image_tensors.clone()
            
            # Randomly add halos between 5 and 7
            n_halos = random.randint(5, 7)

            self.dim1_offset = (image_tensors.shape[1] - self.size) // 2
            self.dim2_offset = (image_tensors.shape[2] - self.size) // 2

            # Add random halos on the image tensors
            for _ in range(n_halos):
                modified_image_tensors = self.add_random_halo(modified_image_tensors)
                # modified_image_tensors = self.add_random_halo(modified_image_tensors)
            # return torch.clamp(modified_image_tensors, max = image_tensors.max())
            return modified_image_tensors
        # torch.clamp(modified_image_tensors, max = image_tensors.max())
            # return torch.clamp(modified_image_tensors, min = torch.amin(image_tensors), max = torch.amax(image_tensors))
        else: return image_tensors

    # Function for adding random halo
    def add_random_halo(self, image_tensor):
        # Randomly generate radius and center points for halo
        self.radius = random.randint(int(0.01 * self.size), int(0.05 * self.size))
        self.center = [random.randint(self.radius + 1, self.size - self.radius - 1),
                        random.randint(self.radius + 1, self.size - self.radius - 1)]
        
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        dist_from_center = torch.sqrt(((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        normalized_dist = dist_from_center / self.radius # Normalize the distance

        # Calculate the intensity of halo effect
        halo_intensity = torch.clamp(self.intensity_range[0] + (self.intensity_range[1] - self.intensity_range[0]) * (1 - normalized_dist), min = 0, max = 1)
        halo_mask = dist_from_center <= self.radius // 2 # Create halo mask
        halo_effect = halo_intensity * (self.radius - dist_from_center) / self.radius # Calculate halo effect
        halo_effect = np.clip(halo_effect, 0, 1) # Clip the effect to valid range
        halo_effect = np.expand_dims(halo_effect, axis = 0)
        halo_effect = np.repeat(halo_effect, image_tensor.shape[0], axis = 0)

        # Add halo to the original image tensor
        image_tensor[:, halo_mask] = image_tensor[:, halo_mask] * (1 - halo_effect[:, halo_mask]) + halo_effect[:, halo_mask]

        # Return the augmented image tensor
        return image_tensor
    

# Augment images by randomly adding halo effects
class Hole(object):
    def __init__(self, size, prob = 0.5):
        self.size = size
        self.prob = prob
        self.center = None
        self.radius = None
        self.tensor_to_image = T.ToPILImage()

    def __call__(self, image_tensors, target = None):
        # Randomly decide whether to add holes based on a probability
        if random.random() < self.prob:
            # Use modified_image_tensors for visualization purpose
            modified_image_tensors = image_tensors.clone()
            
            # Randomly add holes between 5 and 7
            n_halos = random.randint(5, 7)

            self.dim1_offset = (image_tensors.shape[1] - self.size) // 2
            self.dim2_offset = (image_tensors.shape[2] - self.size) // 2

            # Add random holes on the image tensors
            for _ in range(n_halos):
                image_tensors = self.add_random_hole(image_tensors)
                # modified_image_tensors = self.add_random_hole(modified_image_tensors)
            return torch.clamp(image_tensors, min = torch.amin(image_tensors), max = torch.amax(image_tensors))
            # return torch.clamp(modified_image_tensors, min = torch.amin(image_tensors), max = torch.amax(image_tensors))
        else: return image_tensors

    # Function for adding random hole
    def add_random_hole(self, image_tensor):
        # Randomly generate radius and center points for halo
        self.radius = random.randint(int(0.01 * self.size), int(0.05 * self.size))
        self.center = [random.randint(self.radius + 1, self.size - self.radius - 1),
                        random.randint(self.radius + 1, self.size - self.radius - 1)]
        
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        dist_from_center = torch.sqrt(((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        
        hole_mask = dist_from_center <= self.radius // 2 # Create a circular hole mask
        image_tensor[:, hole_mask] = 0 # Apply hole mask to original image tensor

        # Return augmented image tensor
        return image_tensor


# Randomly blur images
class Blur(object):
    def __init__(self, sigma=None, prob = 0.5):
        self.level_range = [0.1, 1] # Range for random blur level
        self.sigma = sigma
        self.prob = prob

    def __get_parameter(self):
        # Get randomly blurred level
        return np.random.uniform(self.level_range[0], self.level_range[1])

    def __call__(self, image, target=None):
        # Randomly apply blurring based on probability
        if random.random() < self.prob:
            # Randomly calcute blur radius width 
            rad_w = random.randint(10, 50)

            # Ensure radius is odd and maintain same ratio with height
            if (rad_w % 2) == 0: rad_w = rad_w + 1 
            rad_h = rad_w

            # Apply Gaussian blur
            image = F.gaussian_blur(image, (rad_w,rad_h))

        # Return blurred image
        return image
    
from PIL import ImageEnhance

# Randomly sharpen the images
def RandomSharpen(image, alpha = 0.2, probability = 0.5):
    # Randomly sharpen image based on probability
    if random.random() <= probability:
        sharpener = ImageEnhance.Sharpness(image) # Create sharpener object
        factor = 0.5  # Sharpening factor
        image = sharpener.enhance(1.0 + alpha * factor) # Enhance the image sharpness
        
        # Return sharpened image
        return image
    else:
        return image


# Crop images from center with particular size
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

        # If no size is given, calculate based on the aspect ratio
        if self.size is None:
            if isinstance(img, torch.Tensor):
                # Get hight and width from tensor
                h, w = img.shape[-2:]
            else:
                # or get width and heigt from image
                w, h = img.size
                
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            
            # Size must match the ratio while cropping to the edge of the image
            # print(ratio, w, h)

            # Calculate the dimensions based on the aspect ratio
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)

            # Determine the size that maintains the aspect ratio and fits within the image
            if w>=h:
                if ratioed_h <= h:
                    # Crop height to fit within the image
                    size = (ratioed_h, w)
                else:
                    # Crop height to fit within the image
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    # Crop height to fit within the image
                    size = (h, ratioed_w)
                else:
                    # Crop height to fit within the image
                    size = (ratioed_h, w)
        else:
            # otherwise, use the specified size if provided
            size = self.size

        # Perform the center crop and return the resized image
        return T.functional.center_crop(img, size)
