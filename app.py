import streamlit as st

st.set_page_config(layout='wide')
st.title('Image Segmentation with YOLOv8')
st.write('Upload an image to perform object detection and segmentation.')
from PIL import Image
import io

st.subheader('Upload your Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image data into a BytesIO object
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data))
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
import os
from ultralytics import YOLO

# Define the path to the model in Google Drive
model_path = 'best.pt'

# Load the YOLOv8 model (ensure this runs only once)
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        model = YOLO(model_path)
        return model
    else:
        st.error(f"Error: Model file not found at {model_path}. Please check your Google Drive path.")
        return None

model = load_model()

if uploaded_file is not None and model is not None:
    if st.button('Run Segmentation'):
        with st.spinner('Performing segmentation...'):
            # Perform inference on the uploaded image
            results = model.predict(source=image, conf=0.25) # Lower confidence threshold for display
            
            # Display results (YOLOv8's plot method returns a PIL Image)
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1]) # RGB PIL image
                st.image(im, caption='Segmentation Result', use_column_width=True)
