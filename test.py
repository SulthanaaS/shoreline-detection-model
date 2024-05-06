import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from hed_unet import HEDUNet
import cv2
import os

# Define the path to the serialized model checkpoint
MODEL_PATH = r"C:\Users\USER\Downloads\HED-UNet-master_timex\HED-UNet-master\logs\2024-05-03_12-16-19\checkpoints\15.pt"

# Define device for inference (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate your HEDUNet model class
model = HEDUNet(input_channels=3)  # Make sure to pass the required input_channels argument

# Load the state dictionary
state_dict = torch.load(MODEL_PATH, map_location=device)

# Filter out keys that don't match
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

# Load the filtered state dictionary
model.load_state_dict(filtered_state_dict, strict=False)

# Move the model to the appropriate device
model = model.to(device) # Set the model to evaluation mode

# Define preprocessing transforms for input images
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize input image to match model input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Function to perform inference on a single image
def predict(image_path):
    # Load and preprocess the input image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # If the model returns multiple outputs, take the first one
    if isinstance(output, tuple):
        output = output[0]
    
    # Post-process the output if needed
    # For example, convert output tensors to numpy arrays, apply thresholding, etc.
    
    return output


def visualize_output(input_image, output):
    # Convert output tensor to numpy array
    output_np = output.squeeze().cpu().numpy()

    # Threshold the output to obtain a binary mask
    threshold = 0.5
    binary_mask = (output_np[0] > threshold).astype(np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours as lines on the input image
    output_image = input_image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  # Draw green lines for contours

    # Display the output image with shoreline overlay
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Path to the folder containing input images
    IMAGES_FOLDER = r"D:\coastal_rushi_timex_new\test\images"

    # Iterate through all images in the folder
    for filename in os.listdir(IMAGES_FOLDER):
        image_path = os.path.join(IMAGES_FOLDER, filename)
        # Load the input image
        input_image = Image.open(image_path).convert("RGB")
        input_image_np = np.array(input_image)

        # Perform inference
        output = predict(image_path)

        # Visualize the output with shoreline overlay
        visualize_output(input_image_np, output)
