import streamlit as st
import torch
from fastai.vision.all import *
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

def load_model():
    """Load the trained MNIST model"""
    return load_learner('mnist_model.pkl')

def preprocess_image(image):
    """Preprocess the drawn image for prediction"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Invert colors (white background to black)
    image = ImageOps.invert(image)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Create a PILImage for FastAI compatibility
    pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    return pil_image

def main():
    st.title('MNIST Digit Recognizer')
    
    # Load the model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Initialize session state for canvas clear
    if 'clear_canvas' not in st.session_state:
        st.session_state.clear_canvas = False
    
    # Create a canvas for drawing
    st.write("Draw a digit in the canvas below:")
    
    # Conditional canvas rendering based on clear state
    if not st.session_state.clear_canvas:
        canvas_result = st_canvas(
            fill_color="#000000",  # Color inside the shapes
            background_color="#FFFFFF",  # Background color of the canvas
            height=280,  # Canvas height
            width=280,  # Canvas width
            drawing_mode="freedraw",  # Free drawing mode
            key="canvas"  # Unique key for the canvas
        )
    else:
        # Render a fresh canvas when cleared
        canvas_result = st_canvas(
            fill_color="#000000",  # Color inside the shapes
            background_color="#FFFFFF",  # Background color of the canvas
            height=280,  # Canvas height
            width=280,  # Canvas width
            drawing_mode="freedraw",  # Free drawing mode
            key="canvas_cleared"  # Different key to force reset
        )
        # Reset the clear state
        st.session_state.clear_canvas = False
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    # Prediction button
    with col1:
        predict_button = st.button('Predict')
    
    # Clear Canvas button
    with col2:
        if st.button('Clear Canvas'):
            st.session_state.clear_canvas = True
            st.experimental_rerun()
    
    # Prediction logic
    if predict_button:
        if canvas_result.image_data is not None:
            # Convert canvas to PIL Image
            input_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            
            # Preprocess the image
            processed_image = preprocess_image(input_image)
            
            # Make prediction
            pred, pred_idx, probs = model.predict(processed_image)
            
            # Display results
            st.write(f"Predicted Digit: {pred}")
            st.write("Confidence Probabilities:")
            
            # Create a bar chart of probabilities
            prob_dict = {str(i): float(prob) for i, prob in enumerate(probs)}
            st.bar_chart(prob_dict)
        else:
            st.warning("Please draw a digit first!")

if __name__ == '__main__':
    main()