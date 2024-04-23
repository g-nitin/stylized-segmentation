import streamlit as st
from PIL import Image
from pathlib import Path
from os.path import exists, join
from stylization import stylize
from utils import combine_with_mask, delete_folder
from subprocess import run

st.set_page_config(
    page_title="Segify: Style",
    page_icon="🎉",
    initial_sidebar_state="expanded"
)


def init():
    Path('./models').mkdir(exist_ok=True)

    # !wget - nc - P. / models / https: // dl.fbaipublicfiles.com / segment_anything / sam_vit_b_01ec64.pth
    url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
    model_file = join(".", "models", "vgg_normalised.pth")

    # Check if the file already exists
    if not exists(model_file):
        run(["wget", "-nc", "-P", "./models/", url])


def delete_and_main(folder_path):
    delete_folder(folder_path)
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    st.switch_page('main.py')


def toggle_styling(boolean):
    st.session_state['perform_styling'] = boolean


st.write(""" # Segify: Style Transfer""")
init()

# Check if the needed variables are available from session state
if 'mask' not in st.session_state:
    st.switch_page('pages/segment.py')

# Get style image logic (similar to segment_page.py)
st.sidebar.subheader('Upload files')
style_image = st.sidebar.file_uploader("## Image for styling:", type=["png", "jpg", "jpeg"])

if style_image:
    style_image_ext: str = style_image.name.split(".")[-1]
    style_image = Image.open(style_image)
    st.session_state['style_image'] = style_image

    toggle_styling(False)
    alpha = st.sidebar.slider("Select styling weight (alpha).\nHigher means more styling.", min_value=0.0, max_value=1.0,
                              value=1.0, on_change=toggle_styling, args=[False])

    # Button to perform stylization
    style = st.sidebar.button("Begin Styling", on_click=toggle_styling, args=[True])
    if style or st.session_state['perform_styling']:
        st.sidebar.markdown('_Styling..._')
        # Output is saved
        stylize(st.session_state['uploaded_image'], style_image,
                output_dir=st.session_state['folder_path'], output_file_name="stylized_mask." + style_image_ext,
                alpha=alpha)

        # Read in image
        temp_style_path: str = join(st.session_state['folder_path'], "stylized_mask." + style_image_ext)
        st.session_state['temp_style_path'] = temp_style_path

        # styled_output_img = ToPILImage()(styled_output.squeeze())  # Don't use
        styled_output_img: Image = Image.open(temp_style_path)

        # Combine images
        combined_image: Image = combine_with_mask(st.session_state['temp_uploaded_path'],
                                                  temp_style_path, st.session_state['mask'])

        # Display the styled output
        st.image(combined_image, caption="Styled Output", use_column_width=True)

        st.sidebar.button('Exit', on_click=delete_and_main, args=[st.session_state['folder_path']])
