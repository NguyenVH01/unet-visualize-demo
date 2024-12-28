import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from unet_model import UNet
from PIL import Image
import torchvision.transforms as transforms

# Set page config and meta tags
st.set_page_config(
    page_title="UNet Architecture Visualization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add HTML meta tags
st.markdown("""
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="UNet Architecture Visualization Tool - Visualize feature maps and network architecture">
        <meta name="keywords" content="UNet, Deep Learning, Image Segmentation, Neural Network, Feature Maps">
        <meta name="author" content="Hoang-Nguyen Vu">
        <meta property="og:title" content="UNet Architecture Visualization">
        <meta property="og:description" content="Interactive tool for visualizing UNet architecture and feature maps">
        <meta property="og:image" content="https://example.com/unet-preview.jpg">
        <meta property="og:type" content="website">
        <title>UNet Architecture Visualization</title>
    </head>
""", unsafe_allow_html=True)

def plot_feature_maps(feature_map, title):
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().numpy()
    
    n_features = min(4, feature_map.shape[1])  # Display up to 4 channels
    fig, axes = plt.subplots(1, n_features, figsize=(15, 3))
    
    if n_features == 1:
        axes = [axes]
    
    for i in range(n_features):
        axes[i].imshow(feature_map[0, i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i+1}')
    
    plt.suptitle(title)
    return fig

def preprocess_image(image):
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def main():
    # Author information at the top
    

    st.title("UNet Architecture Visualization")
    st.write("This app visualizes the basic UNet architecture and its feature maps")
    st.markdown("<h6 style='text-align: left; color: #666666;'>Built by Hoang-Nguyen Vu</h6>", unsafe_allow_html=True)
    st.markdown("---")

    # Model parameters
    in_channels = st.sidebar.slider("Input Channels", 1, 3, 3)
    out_channels = st.sidebar.slider("Output Channels", 1, 3, 1)

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is None:
        st.warning("Please upload an image to visualize the UNet features.")
        return

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Create model
    model = UNet(in_channels=in_channels, out_channels=out_channels)

    # Preprocess the uploaded image
    x = preprocess_image(image)

    # Forward pass with hooks to capture intermediate features
    features = {}
    hooks = []

    def get_features(name):
        def hook(model, input, output):
            features[name] = output
        return hook

    # Register hooks for encoder blocks
    for idx, down in enumerate(model.downs):
        down.register_forward_hook(get_features(f'encoder_{idx+1}'))

    # Register hook for bottleneck
    model.bottleneck.register_forward_hook(get_features('bottleneck'))

    # Register hooks for decoder blocks
    for idx in range(0, len(model.ups), 2):
        model.ups[idx+1].register_forward_hook(get_features(f'decoder_{len(model.ups)//2 - idx//2}'))

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Visualize architecture
    st.header("Network Architecture")
    st.write("""
    The UNet architecture consists of:
    - Encoder path (contracting): 4 blocks of double convolution + max pooling
    - Bottleneck: Double convolution
    - Decoder path (expanding): 4 blocks of upsampling + concatenation + double convolution
    """)

    # Display feature maps
    st.header("Feature Maps Visualization")
    
    # Input
    st.subheader("Input")
    fig = plot_feature_maps(x, "Input Image")
    st.pyplot(fig)

    # Encoder features
    st.subheader("Encoder Features")
    for i in range(len(model.downs)):
        fig = plot_feature_maps(features[f'encoder_{i+1}'], f"Encoder Block {i+1}")
        st.pyplot(fig)

    # Bottleneck
    st.subheader("Bottleneck Features")
    fig = plot_feature_maps(features['bottleneck'], "Bottleneck")
    st.pyplot(fig)

    # Decoder features
    st.subheader("Decoder Features")
    for i in range(len(model.downs)):
        fig = plot_feature_maps(features[f'decoder_{len(model.downs)-i}'], f"Decoder Block {len(model.downs)-i}")
        st.pyplot(fig)

    # Output
    st.subheader("Output")
    fig = plot_feature_maps(output, "Output")
    st.pyplot(fig)

    # Add some space before footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Add footer with author information
    footer = """
    <div class="footer">
        <p style="font-size: 16px; font-weight: 500;">Built by Hoang-Nguyen Vu</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 