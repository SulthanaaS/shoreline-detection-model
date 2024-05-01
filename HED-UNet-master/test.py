import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Function to load the model
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to make predictions
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Main function to run the Streamlit app
def main():
    st.title("Image Classification App")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Display the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Load the model
        model_path = r"C:\Users\USER\Downloads\HED-UNet-master_timex\HED-UNet-master\logs\2024-04-29_10-53-16\checkpoints\01.pt"
        model = load_model(model_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Make prediction
        prediction = predict(preprocessed_image, model)
        
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
