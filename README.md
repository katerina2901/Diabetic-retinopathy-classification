# Diabetic-retinopathy-classification

This repository contains code for Final Project of ML Course in Skoltech.

Install the required packages:
```bash
    pip install -r requirements.txt
```

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
based on retinal images of eyes, which is a chal-
lenging task in itself. Due to the spherical shape
of the eye with unequal distribution of luminosity,
contrast and small lesions, along with the presence
of alike features in healthy and non-healthy retinas,
make it even more difficult. Therefore, we em-
ploy some of the deep learning algorithms/methods
which to achieve this goal i.e., CNN, ViT, and hy-
brid schemes (CNN plus ViT); for example, models
as FasterViT, MedViT, EfficientNet, SwinV2, etc



Running the Application

1. Start the Flask development server:

    ```bash
    python app.py
    ```
2. Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the application.
   - **Upload an Image**: Click the "Choose File" button and select an image of an eye to upload.
   - **Submit the Image**: Click the "Upload" button to submit the image.
   - **View Results**: The application will display the predicted stage of diabetic retinopathy.
