## Segify: Semantic Segmentation for Localized Artistic Effects 

This project tackles segment-based neural style transfer, allowing users to apply artistic styles to specific regions within an image. It combines the efficiency of AdaIN layers with the accuracy of the Segment Anything (SAM) model by Meta AI and offers an interactive user interface for creative exploration.

### Inspiration

Traditional neural style transfer methods apply style globally across the entire image. This project addresses this limitation by enabling users to define the target region for style transfer. This approach is inspired by the following works:

* **Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization** (Huang et al.): This paper introduces AdaIN layers, significantly improving the efficiency of style transfer.
* **Samstyler: Enhancing Visual Creativity with Neural Style Transfer and Segment Anything Model (SAM)** (Psychogyios et al.): This work proposes a system that integrates neural style transfer with the SAM model for user-guided segmentation.

### Approach

Our approach leverages recent advancements in deep learning to achieve real-time, user-guided style transfer for specific image regions:

1. **Real-time Style Transfer with AdaIN:** We incorporate AdaIN layers within a VGG-based style transfer network, enabling efficient style transfer as demonstrated by Huang et al. (2017).
2. **Accurate Segmentation with SAM:** The project utilizes the state-of-the-art SAM model for precise image segmentation, ensuring accurate delineation of the target region for style transfer.
3. **Interactive User Interface:**  A user-friendly interface allows users to:
    * Upload an image.
    * Define a mask to target the specific region for style transfer.
    * Choose the artistic style to apply.
4. **Localized Style Transfer:** The user-defined mask is combined with the AdaIN-powered style transfer model to meticulously apply style only within the designated region.

### Getting Started

However, the following steps show how to run this project locally:

1. **Project Setup:**
   ```python
   git clone https://github.com/g-nitin/stylized-segmentation.git
   cd stylized-segmentation
   pip install -r requirements.txt
   ```
2. **Run:**
   ```python
   streamlit run main.py --server.maxUploadSize 100  # Larger files may cause the app to slow down or quit
   ```

### References

This work was possible by the following papers and implementations:
* Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." Proceedings of the IEEE international conference on computer vision. 2017.
* Kirillov, Alexander, et al. "Segment anything." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
* https://github.com/naoto0804/pytorch-AdaIN/tree/master