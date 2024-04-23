from time import sleep, time
from datetime import timedelta

import streamlit as st
import numpy as np
from typing import List

from segmentation import init, perform_segmentation

st.set_page_config(
    page_title="Segify: Segment",
    page_icon="🎉",
    initial_sidebar_state="expanded"
)

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
st.sidebar.write(f'Segmentation took {timedelta(seconds=time() - start)}')
captions: List[str] = [f'Segment {i + 1}' for i in range(len(segments))]  # Captions for each segment

for i, segment in enumerate(segments):
    # Display each segment
    # Transform to a random color mask
    colored_segment = np.ones((segment.shape[0], segment.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]

    for j in range(3):
        colored_segment[:, :, j] = color_mask[j]

    colored_segment = np.dstack((colored_segment, segment))
    colored_segment = (colored_segment * 255).astype(np.uint8)
    # Image.fromarray(colored_segment).show()
    # Image.fromarray(colored_segment).save(f'output/segment_{i + 1}.png')
    st.image(colored_segment, caption=captions[i])

st.sidebar.subheader("Select segmentation settings")
# st.sidebar.markdown("---")

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

    # Button to navigate to style page
    # next_page = st.sidebar.button("Proceed to Style Transfer")
    # if next_page:
    with st.spinner():
        st.sidebar.markdown("_Switching to the style page_")
        sleep(3)
        st.switch_page('pages/style.py')  # Redirect to style page
