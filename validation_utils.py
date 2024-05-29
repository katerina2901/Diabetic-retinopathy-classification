import os

from datasets import Dataset
from transformers import AutoImageProcessor
from data_utils import CenterCrop
from PIL import Image
import torchvision.transforms as T
import pandas as pd

from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np

from transformers import TrainingArguments
from transformers import Trainer

from sklearn.metrics import confusion_matrix

import torch

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score #, kappa
from sklearn.metrics import roc_auc_score
# from sklearn import metrics
import evaluate

accuracy = evaluate.load("accuracy")

labels = [0, 1, 2, 3, 4]
label2id, id2label = dict(), dict()


def custom_dataset(pretrained_model_name, dataset_name='DDR', resolution = 512):
    if dataset_name == 'DDR':
        labelsTable = pd.read_csv('../DR_grading/DR_grading.csv')
        root_dir = '../DR_grading/DR_grading'
        labelsTable['image_path'] = labelsTable['id_code'].apply(lambda x: os.path.join(root_dir, x))
        labelsTable['label'] = labelsTable['diagnosis']

        test_dataset = labelsTable.drop(columns=['id_code', 'diagnosis'], axis=1)

    model_name_or_path = f'../saved_models/{pretrained_model_name}' 

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    image_processor.size['height'] = resolution
    image_processor.size['width'] = resolution

    _transforms_test = T.Compose([
        CenterCrop(),
    ])

    def load_image(path_image, label, mode):
        # load image
        image = Image.open(path_image)

        image = _transforms_test(image)
        return image

    def func_transform_test(examples):
        inputs = image_processor([load_image(path, lb, 'test').convert("RGB")
                                    for path, lb in zip(examples['image_path'], examples['label'])], return_tensors='pt')
        inputs['label'] = examples['label']
        return inputs

    test_ds = Dataset.from_pandas(test_dataset, preserve_index=False)
    prepared_ds_test = test_ds.with_transform(func_transform_test)
    prepared_ds_test = prepared_ds_test.shuffle(seed=42)

    return prepared_ds_test

def plot_confusion_matrix(y_true, y_pred, file_name,
                          classes=[0, 1, 2, 3, 4],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    np.set_printoptions(precision=2)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f'../results/conf_matrix_{file_name}.png')
    plt.show()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# print("ID2label: ", id2label)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

def get_trainer(model, prepared_ds_test, compute_metrics):
    training_args = TrainingArguments(
        output_dir="./Validation",
        evaluation_strategy="steps",
        logging_steps=50,

        save_steps=50,
        eval_steps=50,
        save_total_limit=3,
        
        remove_unused_columns=False,
        dataloader_num_workers = 16,
        
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        warmup_ratio=0.02,
        
        metric_for_best_model="kappa", 
        greater_is_better = True,
        load_best_model_at_end=True,
        
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds_test,
        eval_dataset=prepared_ds_test
    )

    return trainer


def calculate_per_class_accuracy(confusion_matrix):
        num_classes = confusion_matrix.shape[0]
        per_class_accuracy = []

        for i in range(num_classes):
            TP = confusion_matrix[i, i]
            FN = np.sum(confusion_matrix[i, :]) - TP
            FP = np.sum(confusion_matrix[:, i]) - TP
            TN = np.sum(confusion_matrix) - (TP + FP + FN)

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            per_class_accuracy.append(accuracy)

        return per_class_accuracy

def get_compute_metrics(pretrained_model_name, dataset_name, save_cm=True):

    def compute_metrics(eval_pred):
        save_cm_fn = save_cm
        predictions_proba, labels = eval_pred
        predictions = np.argmax(predictions_proba, axis=1)
        predictions = np.clip(predictions, 0, 4)
        result_accuracy = accuracy.compute(predictions=predictions, references=labels)

        if predictions_proba.shape[1] > 1:  # Check if we have more than one class
            predictions_proba = torch.nn.functional.softmax(torch.tensor(predictions_proba), dim=-1).numpy()

        cm = confusion_matrix(labels, predictions)
        perclass_acc = calculate_per_class_accuracy(cm)

        result = {
                'accuracy': np.mean([result_accuracy['accuracy']]),
                'kappa': np.mean([cohen_kappa_score(labels, predictions, weights = "quadratic")]),
                'f1': np.mean([f1_score(labels, predictions, average='weighted')]),
                # 'roc_auc': np.mean([roc_auc_score(labels, predictions_proba, multi_class='ovr')]),
                'roc_auc': np.mean([roc_auc_score(labels, predictions_proba, multi_class='ovr')]),
                'class_0' : perclass_acc[0],
                'class_1' : perclass_acc[1],
                'class_2' : perclass_acc[2],
                'class_3' : perclass_acc[3],
                'class_4' : perclass_acc[4],
                }
        
        if save_cm:
            plot_confusion_matrix(labels, predictions, normalize=True,
                            title='Normalized confusion matrix', file_name=f'{pretrained_model_name}_{dataset_name}')

        return result

    return compute_metrics