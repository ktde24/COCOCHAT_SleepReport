import sys
import os
import time
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="수면 Report", layout="centered")


# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) # code
chatbot_dir=os.path.dirname(current_dir) # COCOCHAT_sleepreport
data_dir=os.path.join(chatbot_dir, 'Data', 'vectordbbasicparenting_faiss_index')
model_dir = os.path.join(current_dir, 'model')
sys.path.append(model_dir)

icon_path=os.path.join(chatbot_dir, 'Data', 'icon')
pic_path=os.path.join(icon_path, "Dr.COCO.png")
pic2_path = os.path.join(icon_path, "baby.png")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['TOKENIZERS_PARALLELISM'] = st.secrets['TOKENIZERS_PARALLELISM']

import sleep


# Set up your Streamlit application
def setup_streamlit_app():
    # Load and apply CSS for styling (if you have a CSS file for styling)
    style_path = os.path.join(current_dir, "style.css")
    with open(style_path) as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
    
    # Display the header or any initial content
    #st.title("Welcome to the Sleep Report")

# Main function that runs the Streamlit app
def main():
    setup_streamlit_app()

    # Call the main function from the sleep module to display the sleep report
    sleep.main()

if __name__ == "__main__":
    main()