import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from unet_model import UNet
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns

# Enable multi-threading
st.set_page_config(
    page_title="UNet Architecture Visualization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enable multi-threading for better performance
if not hasattr(st, '_is_configured_for_threading'):
    st._is_configured_for_threading = True
    st.experimental_singleton._get_or_create_singleton = lambda *args, **kwargs: None
    st.cache_data.clear()
    st.cache_resource.clear()

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #2c3e50;
        font-weight: 700;
    }
    .stHeader {
        color: #34495e;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .block-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

def plot_feature_maps(feature_map, title, cmap='RdGy_r'):
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().numpy()
    
    n_features = min(4, feature_map.shape[1])
    fig, axes = plt.subplots(1, n_features, figsize=(15, 4))
    fig.patch.set_facecolor('#f0f2f6')
    
    if n_features == 1:
        axes = [axes]
    
    for i in range(n_features):
        # Normalize the feature map
        feat_map = feature_map[0, i]
        vmin, vmax = feat_map.min(), feat_map.max()
        normalized_map = (feat_map - vmin) / (vmax - vmin + 1e-8)
        
        im = axes[i].imshow(normalized_map, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i+1}', pad=10, fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig

def plot_unet_block(feature_maps, block_name, cmap='RdGy_r'):
    """Plot detailed visualization of a UNet block with enhanced styling"""
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    n_channels = min(16, feature_maps.shape[1])
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 3.5*n_rows))
    fig.patch.set_facecolor('#f0f2f6')
    
    # Calculate statistics
    mean_activation = np.mean(feature_maps[0])
    max_activation = np.max(feature_maps[0])
    std_activation = np.std(feature_maps[0])
    
    # Add statistics text
    stats_text = f'Mean: {mean_activation:.2f}\nMax: {max_activation:.2f}\nStd: {std_activation:.2f}'
    fig.text(0.02, 0.98, stats_text, fontsize=10, va='top', bbox=dict(
        facecolor='white', edgecolor='gray', alpha=0.8, pad=5
    ))
    
    # Create grid of subplots
    grid = plt.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    for idx in range(n_channels):
        ax = fig.add_subplot(grid[idx // n_cols, idx % n_cols])
        
        # Normalize the feature map
        feat_map = feature_maps[0, idx]
        vmin, vmax = feat_map.min(), feat_map.max()
        normalized_map = (feat_map - vmin) / (vmax - vmin + 1e-8)
        
        im = ax.imshow(normalized_map, cmap=cmap)
        ax.axis('off')
        ax.set_title(f'Ch {idx+1}\nMax: {vmax:.2f}', fontsize=9)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'{block_name} Feature Maps', fontsize=16, fontweight='bold', y=1.02)
    return fig

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def plot_network_architecture(features, input_shape, output_shape):
    """Plot UNet architecture with feature maps in each block"""
    fig = plt.figure(figsize=(10, 24))  # Taller figure for better spacing
    fig.patch.set_facecolor('white')
    
    # Calculate positions for blocks
    n_encoder_blocks = len([k for k in features.keys() if k.startswith('encoder_')])
    block_width = 0.25     # Wider blocks
    block_height = 0.08    # Taller blocks
    y_spacing = 0.12      # More vertical spacing
    center_x = 0.3        # Left column position
    decoder_x = 0.6       # Right column position
    
    # Create main axes for arrows
    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.axis('off')
    
    # Plot Input at top
    plt.axes([center_x - block_width/2, 0.85, block_width, block_height])
    plt.imshow(features['input'][0, 0], cmap='gray')
    plt.title('Input\n' + str(tuple(input_shape)), pad=10, fontsize=10)
    plt.axis('off')
    
    # Plot Output at top right
    plt.axes([decoder_x - block_width/2, 0.85, block_width, block_height])
    plt.imshow(features['output'][0, 0].detach().cpu(), cmap='gray')
    plt.title('Output\n' + str(tuple(output_shape)), pad=10, fontsize=10)
    plt.axis('off')
    
    # Plot Encoder blocks going down
    for i in range(n_encoder_blocks):
        # Feature map
        y_pos = 0.85 - (i+1)*y_spacing
        ax = plt.axes([center_x - block_width/2, y_pos, block_width, block_height])
        plt.imshow(features[f'encoder_{i+1}'][0, 0].detach().cpu(), cmap='magma')
        plt.title(f'Encoder {i+1}\n{tuple(features[f"encoder_{i+1}"].shape)}', pad=10, fontsize=10)
        plt.axis('off')
        
        # Plot corresponding Decoder block
        ax = plt.axes([decoder_x - block_width/2, y_pos, block_width, block_height])
        plt.imshow(features[f'decoder_{n_encoder_blocks-i}'][0, 0].detach().cpu(), cmap='plasma')
        plt.title(f'Decoder {n_encoder_blocks-i}\n{tuple(features[f"decoder_{n_encoder_blocks-i}"].shape)}', pad=10, fontsize=10)
        plt.axis('off')
        
        # Down arrow
        if i < n_encoder_blocks - 1:
            arrow = FancyArrowPatch(
                (center_x, y_pos - 0.01),
                (center_x, y_pos - y_spacing + 0.02),
                arrowstyle='-|>',
                mutation_scale=20,
                lw=2,
                color='black',
                zorder=10
            )
            main_ax.add_patch(arrow)
        
        # Skip connection (horizontal arrow)
        arrow = FancyArrowPatch(
            (center_x + block_width/2 - 0.02, y_pos + block_height/2),
            (decoder_x - block_width/2 + 0.02, y_pos + block_height/2),
            arrowstyle='-|>',
            mutation_scale=20,
            lw=2,
            color='blue',
            zorder=10
        )
        main_ax.add_patch(arrow)
    
    # Plot Bottleneck at bottom
    bottleneck_y = 0.85 - (n_encoder_blocks+1)*y_spacing
    plt.axes([center_x - block_width/2, bottleneck_y, block_width, block_height])
    plt.imshow(features['bottleneck'][0, 0].detach().cpu(), cmap='inferno')
    plt.title(f'Bottleneck\n{tuple(features["bottleneck"].shape)}', pad=10, fontsize=10)
    plt.axis('off')
    
    # Add title and description
    plt.suptitle('UNet Architecture Overview', fontsize=16, y=0.95)
    plt.figtext(0.02, 0.02, 
                'Blue arrows: Skip connections\nBlack arrows: Feature flow\n' + 
                'Numbers show tensor shapes: (batch, channels, height, width)',
                fontsize=10)
    
    return fig

def main():
    st.title("🔬 UNet Architecture Visualization")
    st.markdown("""
        <div style='background-color: #e8f4f9; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h4 style='margin: 0; color: #2c3e50;'>Interactive Feature Map Visualization</h4>
            <p style='margin: 10px 0 0 0; color: #34495e;'>
                Explore the internal representations learned by each layer of the UNet architecture.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h6 style='text-align: left; color: #666666;'>Built by Hoang-Nguyen Vu</h6>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar styling
    st.sidebar.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>
            <h4 style='margin: 0; color: #2c3e50;'>Model Parameters</h4>
        </div>
    """, unsafe_allow_html=True)
    
    in_channels = st.sidebar.slider("Input Channels", 1, 3, 3)
    out_channels = st.sidebar.slider("Output Channels", 1, 3, 1)

    # Image upload with styled container
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
            <h4 style='margin: 0; color: #2c3e50;'>Upload Image</h4>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image to visualize the UNet features.")
        return

    # Rest of your code remains the same...
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=None)

    model = UNet(in_channels=in_channels, out_channels=out_channels)
    x = preprocess_image(image)

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output
        return hook

    # Register hooks and forward pass remain the same...
    for idx, down in enumerate(model.downs):
        down.register_forward_hook(get_features(f'encoder_{idx+1}'))

    model.bottleneck.register_forward_hook(get_features('bottleneck'))

    for idx in range(0, len(model.ups), 2):
        model.ups[idx+1].register_forward_hook(get_features(f'decoder_{len(model.ups)//2 - idx//2}'))

    with torch.no_grad():
        output = model(x)
        
    # Store input and output in features dict
    features['input'] = x
    features['output'] = output

    

    # Feature maps visualization with enhanced styling
    st.markdown("""
        <div style='background-color: #f5f9f9; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h2 style='color: #2c3e50; margin-bottom: 15px;'>Feature Maps Visualization</h2>
        </div>
    """, unsafe_allow_html=True)

    # Rest of the visualization code remains the same...
    st.subheader("Input Image")
    fig = plot_feature_maps(x, "Input Channels", cmap='gray')
    st.pyplot(fig)

    st.subheader("Encoder Path")
    for i in range(len(model.downs)):
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = plot_feature_maps(features[f'encoder_{i+1}'], f"Encoder Block {i+1} Overview", cmap='magma')
            st.pyplot(fig)
        with col2:
            fig = plot_unet_block(features[f'encoder_{i+1}'], f"Encoder Block {i+1}", cmap='magma')
            st.pyplot(fig)

    st.subheader("Bottleneck")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = plot_feature_maps(features['bottleneck'], "Bottleneck Overview", cmap='inferno')
        st.pyplot(fig)
    with col2:
        fig = plot_unet_block(features['bottleneck'], "Bottleneck", cmap='inferno')
        st.pyplot(fig)

    st.subheader("Decoder Path")
    for i in range(len(model.downs)):
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = plot_feature_maps(features[f'decoder_{len(model.downs)-i}'], f"Decoder Block {len(model.downs)-i} Overview", cmap='plasma')
            st.pyplot(fig)
        with col2:
            fig = plot_unet_block(features[f'decoder_{len(model.downs)-i}'], f"Decoder Block {len(model.downs)-i}", cmap='plasma')
            st.pyplot(fig)

    st.subheader("Output")
    fig = plot_feature_maps(output, "Final Output", cmap='gray')
    st.pyplot(fig)

if __name__ == "__main__":
    main() 