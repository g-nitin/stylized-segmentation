from time import sleep, time
from datetime import timedelta

import streamlit as st
import numpy as np
from typing import List

from segmentation import init, perform_segmentation


def overlay_mask_on_image(image: np.ndarray, segments: List[np.ndarray], blending_factor: float) -> List[np.ndarray]:
    """
    Overlays segmentation masks on an original image for visualization.
    :param image: The original image.
    :param segments: A list of NumPy boolean arrays representing masks.
    :param blending_factor: Blending factor level of the mask overlay (0.0 to 1.0).
    :return: A list of numpy arrays consisting of the original image with overlaid masks.
    """
    result = image.copy().astype(np.uint8)  # Create a copy to avoid modifying the original
    overlays = []

    for segment in segments:
        mask = segment.astype(float) * blending_factor

        # Change the image based on the adjusted mask
        new_im = result * (1 - mask[..., None])

        # Clip the resulting values to stay within 0.0 and 1.0
        new_im = np.clip(new_im, 0.0, 255.0) / 255.0

        overlays.append(new_im)

    return overlays


def main():
    # Check if the needed variables are available from session state
    vars_needed = ['uploaded_image', 'temp_uploaded_path', 'num_masks']
    if not all(var in st.session_state for var in vars_needed):
        st.switch_page('pages/input.py')

    # st.write(""" # Segify: Segmentation""")
    model_file, model_type, device = init()

    # Get the segments as a list
    uploaded_image_np: np.ndarray = np.array(st.session_state['uploaded_image'])
    start = time()

    segments: List[np.ndarray] = perform_segmentation(uploaded_image_np, st.session_state['num_masks'],
                                                      model_file, model_type, device)

    st.sidebar.write(f'Segmentation took {timedelta(seconds= (time() - start))} seconds.')
    captions: List[str] = [f'Segment {i + 1}' for i in range(len(segments))]  # Captions for each segment

    overlays = overlay_mask_on_image(uploaded_image_np, segments, blending_factor=0.77)

    for i, overlay in enumerate(overlays):
        st.image(overlay, caption=captions[i])

    st.sidebar.subheader("Select segmentation settings")
    st.sidebar.markdown("_Segments are highlighted_")

    # Display the selected segment based on user choice on segment_page.py
    selected_seg_num = st.sidebar.selectbox(
        "Select the segment that you want to stylize", tuple(captions), index=None
    )
    if selected_seg_num:
        st.sidebar.write(f'You selected {selected_seg_num}')
        mask_num = int(selected_seg_num.split()[-1]) - 1  # Get the index of the segment

        # The mask of the segmentation to use for stylizing
        mask: np.ndarray = segments[mask_num]
        st.session_state['mask'] = mask

        with st.spinner():
            st.sidebar.markdown("_Switching to the style page_")
            sleep(1)
            st.switch_page('pages/style.py')  # Redirect to style page


if __name__ == "__main__":
    st.set_page_config(
        page_title="Segify: Segment",
        page_icon="🎉",
        initial_sidebar_state="expanded"
    )
    main()
