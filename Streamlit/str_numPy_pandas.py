import streamlit as st
import numpy as np
import pandas as pd
# APP TITLE AND DESCRIPTION 
st.title("NumPy and Pandas in Streamlit")
st.write("This app demonstrates how to use NumPy and Pandas in Streamlit.")
# Interactive widgets in a sidebar
st.sidebar.header("User Input Features")
# Text input
user_input = st.sidebar.text_input("What is your name?", "James Bond")
# Slider
age= st.sidebar.slider("Select your age", 0, 100, 30)
# Selectbox
favorite_color = st.sidebar.selectbox("Select your favorite color", ["Red", "Green", "Blue","Yellow"])
# Main content
st.header("Welcome, " + user_input + "!")
st.write("Your age is: ", age)
st.write("Your favorite color is: ", favorite_color)
# Display data
st.subheader("Here's random data generated using NumPy:")
# Create a sample DataFrame using NumPy
data = pd.DataFrame(
    np.random.rand(10,5),
    columns=('col %d' % i for i in range(5))
)
st.dataframe(data)
# checkboxx to show/hide data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)
#Button to trigger an action
if st.button("Generate New Data"):
    st.write("Generating new data...")
else:  
    st.write("Click the button to generate new data.")


  
    