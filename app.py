import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import os

# Configure TensorFlow to handle GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Function to safely load models and encoders
def load_model_safe():
    try:
        # Check if model file exists
        if not os.path.exists('model.h5'):
            st.error("model.h5 file not found in the current directory")
            return None
        
        # Load model with custom object scope to handle any custom layers
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model('model.h5', compile=False)
            # Recompile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_pickle_safe(filename):
    try:
        if not os.path.exists(filename):
            st.error(f"{filename} not found in the current directory")
            return None
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

# Cache the loading of models and encoders
@st.cache_resource
def load_all_models():
    model = load_model_safe()
    one_hot_encoder = load_pickle_safe('one_hot_encoder.pkl')
    label_encoder_gender = load_pickle_safe('label_encoder_gender.pkl')
    scaler = load_pickle_safe('scaler.pkl')
    
    return model, one_hot_encoder, label_encoder_gender, scaler

def preprocess_input(data, one_hot_encoder, label_encoder_gender):
    try:
        # Handle geography encoding
        geography_data = pd.DataFrame({'Geography': [data['Geography']]})
        geo_encoded = one_hot_encoder.transform(geography_data).toarray()
        
        # Create base dataframe
        input_data = pd.DataFrame({
            'CreditScore': [data['CreditScore']],
            'Gender': [label_encoder_gender.transform([data['Gender']])[0]],
            'Age': [data['Age']],
            'Tenure': [data['Tenure']],
            'Balance': [data['Balance']],
            'NumOfProducts': [data['NumOfProducts']],
            'HasCrCard': [data['HasCrCard']],
            'IsActiveMember': [data['IsActiveMember']],
            'EstimatedSalary': [data['EstimatedSalary']]
        })
        
        # Add geography columns
        countries = ['France', 'Germany', 'Spain']
        for i, country in enumerate(countries):
            input_data[country] = geo_encoded[0][i]
        
        # Ensure correct column order
        columns_order = ['France', 'Germany', 'Spain', 'CreditScore', 'Gender', 'Age', 
                        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
                        'IsActiveMember', 'EstimatedSalary']
        input_data = input_data[columns_order]
        
        return input_data
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def main():
    st.title('Customer Churn Prediction')
    
    # Load all models
    with st.spinner('Loading models...'):
        model, one_hot_encoder, label_encoder_gender, scaler = load_all_models()
    
    # Check if all models loaded successfully
    if None in (model, one_hot_encoder, label_encoder_gender, scaler):
        st.error("Failed to load one or more required components. Please check the error messages above.")
        return
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
            gender = st.selectbox('Gender', ['Male', 'Female'])
            age = st.slider('Age', 18, 92, 30)
            credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
            balance = st.number_input('Balance', min_value=0.0, value=0.0, format="%.2f")
        
        with col2:
            estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=0.0, format="%.2f")
            tenure = st.slider('Tenure (years)', 0, 10, 1)
            num_products = st.slider('Number of Products', 1, 4, 1)
            has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        submit_button = st.form_submit_button("Predict Churn")
    
    if submit_button:
        try:
            # Prepare input data
            input_data = {
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Balance': balance,
                'CreditScore': credit_score,
                'EstimatedSalary': estimated_salary,
                'Tenure': tenure,
                'NumOfProducts': num_products,
                'HasCrCard': has_cr_card,
                'IsActiveMember': is_active_member
            }
            
            # Show processing status
            with st.spinner('Processing...'):
                # Preprocess input
                input_df = preprocess_input(input_data, one_hot_encoder, label_encoder_gender)
                
                if input_df is not None:
                    # Scale features
                    input_scaled = scaler.transform(input_df)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled, verbose=0)
                    prediction_probability = prediction[0][0]
                    
                    # Display results
                    st.subheader('Prediction Results')
                    
                    # Create a gauge-like progress bar
                    st.progress(float(prediction_probability))
                    probability_text = f"Churn Probability: {prediction_probability:.1%}"
                    
                    if prediction_probability > 0.5:
                        st.error(f"⚠️ High Risk of Churn! {probability_text}")
                    else:
                        st.success(f"✅ Low Risk of Churn! {probability_text}")
                    
                    # Display customer profile
                    st.subheader('Customer Profile Summary')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Demographics:**")
                        st.write(f"- Country: {geography}")
                        st.write(f"- Gender: {gender}")
                        st.write(f"- Age: {age}")
                    with col2:
                        st.write("**Account Details:**")
                        st.write(f"- Balance: ${balance:,.2f}")
                        st.write(f"- Credit Score: {credit_score}")
                        st.write(f"- Tenure: {tenure} years")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please check your input values and try again.")

if __name__ == "__main__":
    main()