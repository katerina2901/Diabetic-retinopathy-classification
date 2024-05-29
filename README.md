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

[Here](notebooks/Validation/test.ipynb) one can run validation file with models trained on DDR dataset.


![image](https://github.com/katerina2901/Diabetic-retinopathy-classification/assets/133007241/cab074b4-5a36-4eb0-8357-30f3b36434de)


| Model             | Kappa         | F1    | Accuracy |
| ------------------|:-------------:| -----:|---------:|
| MedViT            | 0.711         | 0.793 | 0.786    |       
| MedViT + attention| 0.637         | 0.734 | 0.729    |
| FasterViT         | ${\color{red}0.774}$         | 0.767 | 0.753    | 
| SwinV2            | 0.751         | ${\color{red}0.806}$ | ${\color{red}0.809}$    | 
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

## 6. Requirements

## 7. References
<a id="1">[1]</a> 
J. W. C. Emma Dugas, Jared, “Diabetic
retinopathy detection,” 2015

<a id="1">[2]</a> 
T. Li, Y. Gao, K. Wang, S. Guo, H. Liu, and
H. Kang, “Diagnostic assessment of deep learn-
ing algorithms for diabetic retinopathy screening,”
Information Sciences, vol. 501, pp. 511 – 522,
2019

<a id="1">[3]</a> 
Chetoui, M. and Akhloufi, M. A. Explain-
able end-to-end deep learning for diabetic retinopa-
thy detection across multiple datasets. J Med Imag-
ing (Bellingham).

<a id="1">[4]</a> 
Huang, Y., Lyu, J., Cheng, P., Tam, R., and
Tang, X. Ssit: Saliency-guided self-supervised im-
age transformer for diabetic retinopathy grading.
IEEE Journal of Biomedical and Health Informat-
ics, 2024.

<a id="1">[5]</a> 
Manzari, O. N., Ahmadabadi, H., Kashiani,
H., Shokouhi, S. B., and Ayatollahi, A. Medvit: A
robust vision transformer for generalized medical
image classification. Computers in Biology and
Medicine, 157:106791, 2023.


<a id="1">[6]</a> 
Sun, R., Li, Y., Zhang, T., Mao, Z., Wu,
F., and Zhang, Y. Lesion-aware transformers for
diabetic retinopathy grading. pp. 10933–10942,
2021.

<a id="1">[7]</a> 
Uysal, E. S., Safak Bilici, M., Zaza, B. S.,
Ozgenc, M. Y., and Boyar, O. Exploring the limits
of data augmentation for retinal vessel segmenta-
tion, 2021.





