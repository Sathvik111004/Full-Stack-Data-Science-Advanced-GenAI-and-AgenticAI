import streamlit as st
import pickle
import numpy as np
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'linear_regression_model.pkl')

# Load the trained model from the pickle file
model = pickle.load(open(model_path, 'rb'))

# Set the title of the app
st.title("Salary Prediction App")
# Description of the app
st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

# Add input widget for years of experience
years_experience = st.number_input("Enter years of experience:", min_value=0.0, max_value=50.0, step=0.1)

# When button is clicked, make prediction and display the result
if st.button("Predict Salary"):
    # Make a prediction using the trained model
    experience_input = np.array([[years_experience]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(experience_input)
   
    # Display the result
    st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.built model by prakash senapati")
