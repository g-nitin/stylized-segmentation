import streamlit as st
from utils import create_folder


def main():
    st.write(""" # Segify: Semantic Segmentation for Localized Artistic Effects  """)

    st.session_state['folder_path'] = create_folder('temp_images')

    st.write("Segify is a user-friendly tool that empowers you to creatively manipulate your images "
             "using the power of neural style transfer.")
    st.write("This app uses multiple pages to navigate you through the styling process. Here's what you can do:")

    st.markdown("**Upload your image:** Upload the image you want to transform.")
    st.markdown("**Target your creativity:** Choose the specific region in your image for localized style transfer.")
    st.markdown("**Choose your style:** Upload the artistic image to infuse with your image.")

    begin = st.button("Segment & Style Images")
    if begin:
        st.switch_page('pages/input.py')

    st.markdown("---")
    st.markdown("_For more information about the app, visit https://github.com/g-nitin/stylized-segmentation_")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Segify",
        page_icon="🎉",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    main()
