# Diabetic-retinopathy-classification

This repository contains code for Final Project of DL Course at Skoltech.

## Repository structure

## Project description

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

## Datasets
Train models on [EyePACS dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview)  - 88k retinal images

Validate models on [DDR dataset](https://github.com/nkicsl/DDR-dataset)  - 14k retinal images


## Results

![image](https://github.com/katerina2901/Diabetic-retinopathy-classification/assets/133007241/cab074b4-5a36-4eb0-8357-30f3b36434de)


| Model             | Kappa         | F1    | Accuracy |
| ------------------|:-------------:| -----:|---------:|
| MedViT            | 0.711         | 0.793 | 0.786    |       
| MedViT + attention| 0.637         | 0.734 | 0.729    |
| FasterViT         | 0.774         | 0.767 | 0.753    | 
| SwinV2            | 0.751         | 0.806 | 0.809    | 
| EffNetb5          | 0.737         | 0.766 | 0.767    |

## Running the Application

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
