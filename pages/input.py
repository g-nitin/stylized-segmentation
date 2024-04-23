import time

import streamlit as st
from PIL import Image
import numpy as np
from os.path import join

st.set_page_config(
    page_title="Segify: Input",
    page_icon="🎉",
    initial_sidebar_state="expanded"
)

# Check if the needed variables are available from session state
if 'folder_path' not in st.session_state:
    st.switch_page('main.py')

st.write(""" # Segify: Inputs""")
st.sidebar.subheader("Upload files")

# Get input image
uploaded_image = st.sidebar.file_uploader("## Image to annotate:", type=["png", "jpg", "jpeg"])
if uploaded_image:
    uploaded_image_ext: str = uploaded_image.name.split(".")[-1]
    uploaded_image = Image.open(uploaded_image)
    uploaded_image_np = np.array(uploaded_image)

    # Add to session state
    st.session_state['uploaded_image'] = uploaded_image

    st.sidebar.markdown("_The top right corner shows the running status of the app._")
    st.sidebar.markdown("---")

    # Save the image to a temp directory for later use
    temp_uploaded_path: str = join(st.session_state['folder_path'], "uploaded." + uploaded_image_ext)
    uploaded_image.save(temp_uploaded_path)
    st.session_state['temp_uploaded_path'] = temp_uploaded_path

    # Get the number of segmented masks to show
    num_masks = st.sidebar.selectbox(
        "Select the maximum number of segmented masks to display",
        [i for i in range(1, 11)]
    )
    st.sidebar.write(f'You selected {num_masks} number of masks.')
    st.session_state['num_masks'] = num_masks

    # Button to begin segmentation
    perform_segment = st.sidebar.button("Begin Segmentation")
    if perform_segment:
        with st.spinner():
            st.sidebar.markdown('_Switching to the segmentation page to show the results..._')
            time.sleep(3)
            st.switch_page('pages/segment.py')
