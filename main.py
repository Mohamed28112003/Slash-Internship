import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# Define transformations for preprocessing the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the MobileNetV2 model pre-trained on ImageNet dataset
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Function to preprocess the image
def load_and_preprocess_image(image):
    img = Image.open(image).convert('RGB')  # Ensure image is in RGB format
    img_tensor = preprocess(img).unsqueeze(0)  # Apply transformations and add batch dimension
    return img_tensor

# Function to decode predictions
def decode_predictions(preds, labels, top=5):
    _, indices = torch.topk(preds, top)
    percentages = torch.nn.functional.softmax(preds, dim=1)[0] * 100
    return [(labels[idx], '{:.2f}%'.format(percentages[idx].item())) for idx in indices[0]]

# Streamlit UI
st.title('Image Component Analyzer Using MobileNetV2 ')
st.write("Upload an image and click 'Analyse Image' to detect components.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Check if the 'Analyse Image' button is clicked
    if st.button('Analyse Image'):
        # Preprocess the uploaded image
        img_tensor = load_and_preprocess_image(uploaded_file)

        # Predict with the model
        with torch.no_grad():
            preds = model(img_tensor)

        # Fetch ImageNet labels
        labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')

        # Decode predictions
        decoded_preds = decode_predictions(preds, labels, top=5)

        # Display the results
        st.write("### Predicted Components:")
        for i, (label, score) in enumerate(decoded_preds):
            st.write(f"{i + 1}: {label} (score: {score})")
