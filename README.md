# TLBass
Training the model and the fusion model

This project contains two core programs for training individual foundation models and for model fusion (ensemble). Below is a description of each program and instructions for use.


1.1 Single Model Training and Evaluation (trainAndEvaluateModel.m)
Function:
This script is used to train and evaluate a single foundation model. The inputs are pre-extracted features and their corresponding labels.
Inputs:
features: The feature matrix (each row is a sample, each column is a feature).
labels: The label vector corresponding to each sample.
Outputs:
Trained model parameters.
Evaluation results (e.g., accuracy, loss).
Usage:
Prepare the input features and labels.
Run trainAndEvaluateModel.m in MATLAB and provide the necessary data as prompted.


1.2 Model Fusion and Weight Optimization (Fusion_model_TLBass.m)
Function:
This script is used for fusing (ensembling) three foundation models by optimizing the fusion weights to improve overall performance. It outputs the optimal weights and the fused model's results.
Inputs:
Predictions or output results from three foundation models.
Outputs:
Optimized fusion weights.
Fused model predictions and evaluation metrics.
Usage:
First, use the three foundation models to predict on the same dataset and save their prediction results.
Run Fusion_model_TLBass.m in MATLAB, providing the three sets of model predictions as input.
The script will automatically optimize the fusion weights and output the results.


File Structure
├── trainAndEvaluateModel.m # Single foundation model training and evaluation
├── Fusion_model_TLBass.m # Fusion model weight optimization
├── README.md # This instruction file
└── ... # Other auxiliary files/data

Requirements
MATLAB 2016 or newer
Relevant Machine Learning/Deep Learning Toolboxes installed
How to Run
Prepare the required input data (features, labels, model predictions, etc.) as described above.
Run trainAndEvaluateModel.m and Fusion_model_TLBass.m in sequence, following the script instructions.
Contact
For any questions, please contact the project maintainer or refer to the comments within the code.
