# Diabetic-retinopathy-classification

This repository contains code for Final Project of DL Course at Skoltech.

## 1. Repository structure

```bash
Project
├── app # folder to run web page
│   ├── app.py # file to run web page
│   └── requirements.txt # requirements to run app.py
├── results #  folder with plots
├── README.md



├── results # results of expirements
│   ├── 1st_experiment_base_estimators.csv # contain 1st_experiment best results for all datasets
│   ├── 2nd_experiment_preprocessors.csv # contain 2st_experiment all results for all datasets
│   ├──  3rd_experiment-preprocessors_all.zip # contain 3rd_experiment all results for all datasets
│   ├── baseline_best_algorithms.csv # contain baseline best results for all datasets
│   ├── baseline_experiment.csv # contain baseline results for models without hyperparameter tuning
│   ├── best_algorithm_preprocessor_summary.csv # contain 3 st_experiment best results for all datasets
│   └── final_result.csv # contain a comparison of the best results from all experiments for all datasets
├── README.md
└── dataset_loader.py # function for loading dataset from UCI Machine Learning Repository
```
## 2. Project description

Diabetic retinopathy is a serious complication of 
diabetes that can lead to visual impairment and
blindness in some diabetic patients. It is essential
to identify retinopathy in diabetic patients as soon
as possible to reduce the risks of complications
and preserve vision. An artificial model capable
of differentiating, healthy and non-healthy eyes,
and determining the severity of the disease with
high reliability is highly awaited. Therefore, the
main goal of this study is to develop an effective
approach to classify diabetic retinopathy in patients
based on retinal images of eyes, which is a challenging task in itself. Due to the spherical shape
of the eye with unequal distribution of luminosity,
contrast and small lesions, along with the presence
of alike features in healthy and non-healthy retinas,
make it even more difficult. Therefore, we employ some of the deep learning algorithms/methods
which to achieve this goal i.e., CNN, ViT, and hybrid schemes (CNN plus ViT); for example, models
as FasterViT, MedViT, EfficientNet, SwinV2, etc

### 2.1 [Preprocessing](data_utils.py) 

The preprocessing steps involved several stages:
  * __Data Resampling__: Datasets are highly imbalanced; thus resampling techniques were
applied. The final value of each class is 20%
  * __Resizing Images__
  * __Normalization__
  * __Augmentation__: Data augmentation techniques as sharpness, white Gaussian noise, blurring, flipping, and zooming were applied to enhance model’s ability to generalize from the training data
  * __Post Transform Augmentation__: To enhance
the model’s robustness to classify healthy or
disease severity level on unseen dataset, we
considered randomly adding spots, halos, and
holes on images (after transforms).

### 2.2 Models
The following models were implemented:
  * __FasterViT and [MedViT](https://github.com/Omid-Nejati/MedViT)__: Both models have
been successfully implemented and trained on
the preprocessed dataset. Models’ attention
mechanisms help in identifying key features
in the retinal images. Besides, trying to enhance MedViT model, we experimented putting Channel Attention instead of original one to see how it effects on preformance.
  * __SwinV2__: The Swin Transformer has been implemented to take advantage of its hierarchical
feature representation, which is particularly
beneficial for processing high-resolution images.
  * __EfficientNet__: This model is being explored
to utilize its efficient scaling and performance
capabilities


### 2.3 Metrics and the Loss Function
To evaluate the models’ performance, we utilized
Kappa score, F1 score, accuracy, and per class
accuracy metrics.
The CrossEntropyLoss was utilized as loss function.

### 2.4 Classification Strategies

Various classification strategies were employed,
such as follows:
  * __Multi-class classification.__
  * __Two stage classification:__
    * __Primary Classifier (One vs. All):__ Dis-
tinguishes between healthy (class 0) and
unhealthy (classes 1, 2, 3, 4) eyes.
    * __Secondary Classifier (All vs. All):__ For
images classified as unhealthy, a second
classifier predicts a specific level of dia-
betic retinopathy.
  * __Multistep classification:__ Repeat One vs. All
for all classes, i.e. class 0 vs. classes 1,2,3,4,
then class 1 vs. classes 2,3,4, etc.




## 3. Datasets
Train models on [EyePACS dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview)  - 88k retinal images

Validate models on [DDR dataset](https://github.com/nkicsl/DDR-dataset)  - 14k retinal images

## 4. Results

![image](https://github.com/katerina2901/Diabetic-retinopathy-classification/assets/133007241/cab074b4-5a36-4eb0-8357-30f3b36434de)


| Model             | Kappa         | F1    | Accuracy |
| ------------------|:-------------:| -----:|---------:|
| MedViT            | 0.711         | 0.793 | 0.786    |       
| MedViT + attention| 0.637         | 0.734 | 0.729    |
| FasterViT         | 0.774         | 0.767 | 0.753    | 
| SwinV2            | 0.751         | 0.806 | 0.809    | 
| EffNetb5          | 0.737         | 0.766 | 0.767    |

## 5. Running the Application

Install the required packages:
```bash
    pip install -r requirements.txt
```

1. Start the Flask development server:

    ```bash
    python app.py
    ```
2. Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the application.
   - **Upload an Image**: Click the "Choose File" button and select an image of an eye to upload.
   - **Submit the Image**: Click the "Upload" button to submit the image.
   - **View Results**: The application will display the predicted stage of diabetic retinopathy.

## 5. Requirements




