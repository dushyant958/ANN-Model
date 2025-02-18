# **Bank Customer Churn Prediction using Artificial Neural Networks**

## **Project Overview**
This project implements a deep learning model to predict customer churn for a bank. Using an **Artificial Neural Network (ANN)**, we analyze various customer attributes to predict whether a customer is likely to exit their relationship with the bank. This binary classification problem helps banks identify customers who might leave, enabling **proactive retention measures**.

## **Dataset Description**
The dataset contains various customer attributes, including:
- **Customer ID and basic demographics** (Gender, Age, Geography)
- **Banking relationship information** (Tenure, Balance, Number of Products)
- **Customer engagement metrics** (HasCrCard, IsActiveMember)
- **Financial indicators** (CreditScore, EstimatedSalary)
- **Target variable:** `Exited` (0 = Retained, 1 = Churned)

## **Features**
✅ Data preprocessing and cleaning  
✅ Exploratory Data Analysis (EDA)  
✅ Feature scaling and encoding  
✅ Implementation of Artificial Neural Network  
✅ Model evaluation and performance metrics  
✅ Prediction functionality  

## **Installation & Requirements**
To run this project, install the required dependencies:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

## **Model Architecture**
- **Input Layer:** Matches the number of features after preprocessing  
- **Hidden Layers:** Multiple dense layers with **ReLU activation**  
- **Output Layer:** Single neuron with **sigmoid activation** for binary classification  
- **Optimization:** Adam optimizer  
- **Loss Function:** Binary Cross-Entropy  

## **Robust Code Implementation in `app.py`**
The `app.py` file contains a **well-structured and efficient** implementation of the churn prediction model. It includes:
- ✅ **Automated data preprocessing** to handle missing values and encode categorical features.
- ✅ **A robust ANN model** built with TensorFlow/Keras, ensuring optimal performance.
- ✅ **Efficient model training and evaluation** with metrics like accuracy, precision, recall, and F1-score.
- ✅ **User-friendly prediction functionality**, allowing real-time predictions on new customer data.
- ✅ **Logging and error handling mechanisms** to ensure smooth execution.

## **Usage**
1. Clone this repository:
```bash
git clone https://github.com/your-username/bank-churn-ann.git
cd bank-churn-ann
```
2. Run the `app.py` script:
```bash
python app.py
```
3. Input new customer data to get churn predictions.

## **Future Improvements**
🔹 Implement feature importance analysis  
🔹 Try different neural network architectures  
🔹 Add cross-validation  
🔹 Experiment with different sampling techniques for imbalanced classes  
🔹 Implement hyperparameter tuning  

## **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
🚀 **Feel free to contribute, suggest improvements, or star the repository!**
