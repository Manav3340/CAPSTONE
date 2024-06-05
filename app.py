import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model_path = 'yolov8n.pt'  # Update this path to your best.pt or last.pt model file
model = YOLO(model_path)

# Streamlit app title
st.title("YOLOv8 Object Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Perform object detection
    st.write("Detecting objects...")
    results = model(image)  # Pass the image to the model

    # Convert results to dictionary
    detections = results.pandas().xyxy[0]  # Get the dataframe of results
    st.write(detections)  # Display the detections

    # Draw boxes on the image
    image_np = np.array(image)
    for idx, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        # Draw bounding box
        image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label near bounding box
        image_np = cv2.putText(image_np, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to PIL image and display
    result_image = Image.fromarray(image_np)
    st.image(result_image, caption='Processed Image with Detections.', use_column_width=True)
