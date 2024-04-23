import streamlit as st
from utils import create_folder

st.set_page_config(
    page_title="Segify",
    page_icon="🎉",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.write(""" # Segify: Semantic Segmentation for Localized Artistic Effects  """)

st.session_state['folder_path'] = create_folder('temp_images')


st.write("SamStyler is a user-friendly tool that empowers you to creatively manipulate your images "
         "using the power of neural style transfer.")
st.write("This app uses multiple pages to navigate you through the styling process. Here's what you can do:")

st.markdown("**Upload your image:** Upload the image you want to transform.")
st.markdown("**Target your creativity:** Choose the specific region within your image for localized style transfer.")
st.markdown("**Choose your style:** Upload the artistic image to infuse with your image.")

begin = st.button("Segment & Style Images")
if begin:
    st.switch_page('pages/input.py')
