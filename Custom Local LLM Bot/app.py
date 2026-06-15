import streamlit as st
from ollama import Client

# Create client instance (very important)
client = Client(host="http://localhost:11434")
st.set_page_config(
    page_title="LLM model by GS - OLLAMA",
    layout="centered",
)
st.title("Mr. GS - OLLAMA APP")
prompt = st.text_area("Enter your prompt here:", height=200)

if st.button("Generate Response"):
    if prompt.strip()=="deepseek-r1:1.5b":
        st.warning("This is a special prompt. Please enter a different one.")
    else:
        with st.spinner("Thinking..."):
            response=client.chat(
                model="deepseek-r1:1.5b",
                messages=[{"role": "user", "content": prompt}]
            )
            st.success("Response generated!")
            st.subheader("Response:")
            st.write(response["message"]["content"])   


