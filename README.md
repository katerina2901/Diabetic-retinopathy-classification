# Diabetic-retinopathy-classification

This repository contains code for Final Project of DL Course at Skoltech.

## Repository structure

Project
├── app.py # File for running application
├── Experiments 
│   ├── baseline.py # repeat experiments from main paper with boosting methods
│   ├── 1st_experiment_base_estimators.py # experiments with different base estimators
│   ├── 2st_experiment_preprocessors.py # experiments with different preprocessing operators
│   └── 3st_experiment_resampling.py # experiments with using data-level approach resampling methods
├── Metrics 
│   ├── all_metrics_append.py # function for evaluating all metrics
│   └── all_metrics_tables.py # function for pritning metrics in tables
├── Datasets # Dataset from UCI Machine Learning Repository that can't be imported in python
│   ├── hayes_roth.data
│   ├── new-thyroid.data
│   ├── page_blocks.data
│   ├── shuttle.trn
│   ├── shuttle.tst
│   └── vertebral.dat
├── report # report deliverables
│   ├── presentation.pdf
│   └── report.pdf
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
