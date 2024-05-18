import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset
from torchvision import transforms as T
from transformers import ViTImageProcessor


dataset_folder_name = '../mnt/local/data/kalexu97/large_dataset'
# dataset_folder_name = r"D:\Study\M.E. IoTWT\5. Term 4\Deep Learning\Project\short_imbalanced_dataset"

def load_dataset_path2images(dataset_folder_name):
    train_test_folders = os.listdir(dataset_folder_name)
    datasets = {}
    for trts_split in train_test_folders:
        class_folders = os.listdir(dataset_folder_name+'/'+trts_split)
        # class_folders = os.listdir(dataset_folder_name + '\\' + trts_split)
        labels = []
        paths = []
        for class_folder in class_folders:
            image_names = os.listdir(dataset_folder_name+'/'+trts_split+'/'+class_folder)
            image_paths = [dataset_folder_name+'/'+trts_split+'/'+class_folder+'/'+x for x in image_names]
            # image_names = os.listdir(dataset_folder_name + '\\' + trts_split + '\\' + class_folder)
            # image_paths = [dataset_folder_name + '\\' + trts_split + '\\' + class_folder + '\\' + x for x in image_names]
            class_labels = [int(class_folder)] * len(image_paths)
            labels.extend(class_labels)
            paths.extend(image_paths)
        local_dataset = {'image_path' : paths, 'label' : labels}
        datasets[trts_split] = pd.DataFrame.from_dict(local_dataset)

    return datasets

dataset = load_dataset_path2images(dataset_folder_name)

# Datasets containing paths and labels
train_dataset = dataset['train']
test_dataset = dataset['test']

# Randomly droping samples from majority class upto 0.2*len(majority class)
majority_class = train_dataset[train_dataset['label'] == 0]
drop_indices = np.random.choice(majority_class.index, size = int(majority_class.shape[0] - 0.7 * majority_class.shape[0]), replace = False)
train_dataset = train_dataset.drop(drop_indices)

# oversampling just repeating minority class items enough times to be equal to major dataset in size
max_size = train_dataset['label'].value_counts().max()
lst = [train_dataset]
for class_index, group in train_dataset.groupby('label'):
    lst.append(group.sample(max_size - len(group), replace = True))
train_dataset = pd.concat(lst)

labels = set(train_dataset.label.values)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Processor Checkpoints
model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

transform_plus_rand = T.Compose([
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomVerticalFlip(p = 0.5),
    T.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
])

def load_image(path_image, label, mode):
    # load image
    image = Image.open(path_image)

    if mode == 'train' and label != 0:
        image = transform_plus_rand(image)
        return image
    else:
        return image


# MyDataset class for preprocessing
class MyDataset():
    def __init__(self, dataset, processor, mode = None):
        self.dataset = dataset
        self.processor = processor 
        self.mode = mode

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        if self.mode == 'train':
            return self.dataset.with_transform(self.func_transform_train)
        elif self.mode == 'test':
            return self.dataset.with_transform(self.func_transform_test)

    def func_transform_train(self, examples):
        loaded_images = [load_image(path, label, self.mode) for path, label in zip(examples['image_path'], examples['label'])]
        inputs = self.processor(loaded_images, return_tensors = 'pt')
        inputs['labels'] = examples['label']
        return inputs

    def func_transfrom_test(self, examples):
        loaded_images = [load_image(path, label, self.mode) for path, label in zip(examples['image_path'], examples['label'])]
        inputs = self.processor(loaded_images, return_tensors = 'pt')
        inputs['labels'] = examples['label']
        return inputs


# Create pandas datasets for image paths and labels
train_ds = Dataset.from_pandas(train_dataset, preserve_index = False)
test_ds = Dataset.from_pandas(test_dataset, preserve_index = False)

# Image tensors and labels datasets
prepared_ds_train = prepared_train = MyDataset(train_ds, processor, mode = 'train')[0]
prepared_ds_test = prepared_train = MyDataset(test_ds, processor, mode = 'test')[0]
