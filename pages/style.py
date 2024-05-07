import streamlit as st
from PIL import Image
from os.path import join
from stylization import stylize
from utils import combine_with_mask, delete_folder, get_model


def init():
    """Initialize the models."""

    model_dir = 'models'
    model_name = "vgg_normalised.pth"
    url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"

    response = get_model(model_name,
                         url,
                         model_dir)

    if not response[0]:  # Check if the file downloaded successfully
        print(response[1])
        exit()


def delete_and_main(folder_path):
    """Delete the temp folder and keys in the session state. Also, switch to the main page."""
    delete_folder(folder_path)
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    st.switch_page('main.py')


def toggle_styling(boolean):
    st.session_state['perform_styling'] = boolean


def main():
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
        alpha = st.sidebar.slider("Select styling weight (alpha).\nHigher means more styling.", min_value=0.0,
                                  max_value=1.0,
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

            # styled_output_img: Image = Image.open(temp_style_path)

            # Combine images
            combined_image: Image = combine_with_mask(st.session_state['temp_uploaded_path'],
                                                      temp_style_path, st.session_state['mask'])

            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state['uploaded_image'], use_column_width=True, caption='Original')
            with col2:
                st.image(combined_image, use_column_width=True, caption='Styled')

            st.sidebar.button('Exit', on_click=delete_and_main, args=[st.session_state['folder_path']])


if __name__ == "__main__":
    st.set_page_config(
        page_title="Segify: Style",
        page_icon="🎉",
        initial_sidebar_state="expanded",
        layout='wide'
    )
    main()
