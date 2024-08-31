# Accident Prediction Model

This repository contains the implementation and training process for an accident prediction model. The model uses a deep learning approach to predict accidents in video sequences with high precision and recall.

## Overview

The model is trained to predict the time to accidents in video clips. It utilizes a combination of state-of-the-art techniques and architectures to achieve high accuracy. The training process involves multiple epochs, with the model showing rapid convergence and consistent improvement in performance.

## Model Details

- *Total Trainable Parameters*: 3,581,698
- *Model Architecture*: Based on a combination of deep learning techniques with custom modifications for accident prediction.
- *Training Dataset*: A dataset consisting of labeled video clips where accidents are annotated.

## Training Results

The model was trained over several epochs, with significant improvements observed:

  - Best Frame Average Precision: 99.91%

The model demonstrated rapid improvement and achieved near-perfect precision, making it highly effective for accident prediction.

## Usage

To use this model, follow these steps:

1. *Clone the Repository*:
   ``
   git clone git@github.com:Early-Accident-Anticipation-Graph/Graph_Code.git
   cd Graph_Code
   

2. *Install Dependencies*:
   Make sure you have all necessary dependencies installed. You can install them using:
   ``
   pip install -r requirements.txt
   
## Conclusion

This accident prediction model achieves high accuracy in predicting accidents within video sequences. The training results show consistent improvement, with the model converging rapidly to a high-performance state. This model can be further fine-tuned and adapted for various real-world applications involving accident detection and prevention.

## License

This project is licensed under the MIT License.
