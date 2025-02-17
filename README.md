# Bank Customer Churn Prediction using Artificial Neural Networks

## Project Overview
This project implements a deep learning model to predict customer churn for a bank. Using an Artificial Neural Network (ANN), we analyze various customer attributes to predict whether a customer is likely to exit their relationship with the bank. This binary classification problem helps banks identify customers who might leave, enabling proactive retention measures.

## Dataset Description
The dataset contains various customer attributes including:
- Customer ID and basic demographics (Gender, Age, Geography)
- Banking relationship information (Tenure, Balance, Number of Products)
- Customer engagement metrics (HasCrCard, IsActiveMember)
- Financial indicators (CreditScore, EstimatedSalary)
- Target variable: 'Exited' (0 = Retained, 1 = Churned)

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature scaling and encoding
- Implementation of Artificial Neural Network
- Model evaluation and performance metrics
- Prediction functionality

## Requirements
```
numpy
pandas
tensorflow
scikit-learn
matplotlib
seaborn
```

## Model Architecture
- Input Layer: Matches the number of features after preprocessing
- Hidden Layers: Multiple dense layers with ReLU activation
- Output Layer: Single neuron with sigmoid activation for binary classification
- Optimization: Adam optimizer
- Loss Function: Binary Cross-Entropy

## Future Improvements
- Implement feature importance analysis
- Try different neural network architectures
- Add cross-validation
- Experiment with different sampling techniques for imbalanced classes
- Implement hyperparameter tuning

## License
This project is licensed under the MIT License - see the LICENSE file for details
