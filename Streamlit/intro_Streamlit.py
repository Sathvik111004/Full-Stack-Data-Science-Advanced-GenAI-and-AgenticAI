# st.tile() - to set the title of the app
# st.write() - to write text and display variables
# st.header() - to create a header for a section
# st.slider() - to create a slider for user input


# 1. Import Streamlit
import streamlit as st

# 2. Write a title and some text
st.title("Hello, Streamlit!")

# 3. Add some text
st.write("This is a simple Streamlit app to demonstrate how to use Streamlit for building interactive web applications.")

# 4. Add a slider
st.header("Select a number")
number = st.slider("Choose a number", 0, 100, 50) # min 0, max 100, default value 50

# 5. Display the selected number
st.header("You selected:")
squared = number ** 2
st.write(f"The square of {number} is {squared}.")

