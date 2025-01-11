import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self):
        """Initialize the image classifier with EfficientNetB7 model."""
        try:
            self.model = EfficientNetB7(weights='imagenet')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error("Failed to load the classification model. Please try again later.")
            
    def preprocess_image(self, img: np.ndarray, target_size: Tuple[int, int] = (600, 600)) -> np.ndarray:
        """
        Preprocess the image for model input.
        
        Args:
            img: Input image array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            return preprocess_input(img)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def classify_image(self, image_path: str) -> Optional[Tuple[str, float]]:
        """
        Classify the image and return prediction with confidence score.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_label, confidence_score) or None if error occurs
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
                
            processed_img = self.preprocess_image(img)
            preds = self.model.predict(processed_img, verbose=0)
            prediction = decode_predictions(preds, top=1)[0][0]
            return prediction[1], float(prediction[2])  # Return label and confidence
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            return None

@st.cache_resource
def get_classifier() -> ImageClassifier:
    """Create or retrieve cached classifier instance."""
    return ImageClassifier()

def create_temp_file(uploaded_file) -> str:
    """Create a temporary file from uploaded content."""
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error creating temporary file: {str(e)}")
        raise

def main():
    st.set_page_config(
        page_title="PetPals Image Classifier",
        page_icon="🐾",
        layout="centered"
    )

    st.title("🐾 PetPals Image Classification")
    st.write("""
    Welcome to **PetPals**, your intelligent assistant for image classification. Upload an image, and our AI will identify its content.
    """)

    # Initialize classifier
    classifier = get_classifier()

    # Initialize session state for result visibility
    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    # File uploader with clear instructions
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit image for best results"
    )

    if uploaded_file is not None:
        try:
            # Display image with loading spinner
            with st.spinner("Loading image..."):
                image = Image.open(uploaded_file)
                
                # Resize the image to make it smaller
                image.thumbnail((400, 400))  # Resize to fit within 400x400 pixels
                st.image(image, caption="Uploaded Image", use_container_width=False, width = 200)

            # Classification button with progress tracking
            if st.button("🔍 Classify Image"):
                with st.spinner("Analyzing image..."):
                    # Create temporary file
                    temp_path = create_temp_file(uploaded_file)

                    # Get prediction
                    result = classifier.classify_image(temp_path)

                    # Clean up temporary file
                    os.unlink(temp_path)

                    if result:
                        label, confidence = result
                        st.session_state.show_result = True

                        # Display classification result
                        if st.session_state.show_result:
                            st.subheader("Classification Result")
                            st.write(f"**Label:** {label.title()}")
                            st.write(f"**Confidence:** {confidence * 100:.1f}%")

                            # Hide result button
                            if st.button("Hide Result"):
                                st.session_state.show_result = False

                        # Feedback bar
                        st.progress(confidence)
                        if confidence < 0.5:
                            st.warning("⚠️ Low confidence prediction. Consider uploading a clearer image.")

                    else:
                        st.error("Failed to classify image. Please try again.")

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            st.error("An error occurred while processing the image. Please try again.")

    # Add tips at the bottom
    with st.expander("ℹ️ Tips for Better Results"):
        st.write("""
        - Upload clear, well-lit images.
        - Ensure the subject is centered in the frame.
        - Avoid blurry or very dark images.
        - Supported formats: JPG and PNG.
        """)


if __name__ == "__main__":
    main()