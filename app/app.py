import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import sys
import os

# Set up the path for importing models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.fusion_model import MultiModalDiagnosisModel
from utils.validator import validate_image_upload

# Configuration
IMG_SIZE = 64
N_CLASSES = 4
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
CLASSES = ['COVID', 'Normal', 'Lung Opacity', 'Viral Pneumonia']

# Load Model with cache to prevent redundant loading
@st.cache_resource
def load_trained_model(model_path, device='cpu'):
    """
    Initialize and load the trained multi-modal diagnosis model.
    """
    model = MultiModalDiagnosisModel(
        img_size=IMG_SIZE, patch_size=8, in_channels=1, img_embed_dim=128, metadata_dim=5, n_classes=4
    )
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()
    return model

# Image Processing
def preprocess_image(pil_img):
    """
    Preprocess image as per training (64x64, grayscale, normalize).
    """
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(pil_img).unsqueeze(0) 

# Streamlit App UI
def main():
    st.set_page_config(page_title="ViT Multi-Modal Medical Diagnosis System", layout="wide")
    
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction-card {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .prediction-text {
            font-size: 1.5rem;
            color: #1a1a1a;
            margin-bottom: 10px;
        }
        .prediction-val {
            font-size: 2.5rem;
            font-weight: 900;
            color: #2e7d32;
        }
        </style>
        <div class="main-header">   
            NeuroSyn
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload X-ray Scan")
        uploaded_file = st.file_uploader("Choose an X-ray image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
        
        is_val_xray = False
        image = None
        if uploaded_file is not None:
            is_val_xray, message = validate_image_upload(uploaded_file)
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            try:
                image = Image.open(uploaded_file)
            except Exception:
                image = None

            if not is_val_xray:
                st.warning(f"⚠️ **Rejected:** {message}")
                if image is not None:
                    st.image(image, caption="Rejected Upload", width="stretch")
            else:
                st.success(f"✅ **X-ray Validated:** {message}")
                if image is not None:
                    st.image(image, caption="Validated X-ray Scan", width="stretch")
        else:
            st.info("Please upload an X-ray image to begin.")

    with col2:
        st.subheader("👤 Patient Metadata & Diagnosis")
        
        # User Metadata Input
        age = st.slider("Patient Age", 1, 100, 45)
        gender = st.selectbox("Patient Gender", ["Male", "Female"])
        
        st.write("**Patient Symptoms:**")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            fever = st.checkbox("Fever")
        with col_s2:
            cough = st.checkbox("Cough")
        with col_s3:
            sob = st.checkbox("Shortness of Breath")
        
        if st.button("🚀 Run Diagnosis", width="stretch"):
            if uploaded_file is None:
                st.error("Missing input X-ray image. Please upload a scan.")
            elif not is_val_xray:
                st.error("Cannot run diagnosis: The uploaded image did not pass X-ray validation. Please upload a valid medical radiography scan.")
            elif image is None:
                st.error("Unable to read image. Please upload a valid PNG/JPG/JPEG chest X-ray image.")
            else:
                with st.spinner("Processing multi-modal features..."):
                    try:
                        # Prepare data
                        img_tensor = preprocess_image(image)
                        
                        # Normalize metadata
                        age_norm = float(age) / 100.0
                        gender_encoded = 1.0 if gender == "Female" else 0.0
                        fever_val = 1.0 if fever else 0.0
                        cough_val = 1.0 if cough else 0.0
                        sob_val = 1.0 if sob else 0.0
                        
                        # Metadata Tensor: [Age, Gender, Fever, Cough, SOB]
                        meta_tensor = torch.tensor([[age_norm, gender_encoded, fever_val, cough_val, sob_val]], dtype=torch.float32)
                        
                        # Load model and predict
                        model = load_trained_model(MODEL_PATH)
                        with torch.no_grad():
                            outputs = model(img_tensor, meta_tensor)
                            _, predicted = torch.max(outputs, 1)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            
                        # Show Result
                        result_idx = predicted.item()
                        confidence = probabilities[0][result_idx].item() * 100
                        result_class = CLASSES[result_idx]
                        
                        st.markdown(f"""
                            <div class="prediction-card">
                                <div class="prediction-text">Diagnosis Result:</div>
                                <div class="prediction-val">{result_class}</div>
                                <div class="prediction-text">Confidence: {confidence:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Simple Bar Chart for all classes
                        st.write("Full probabilities:")
                        probs_dict = {CLASSES[i]: float(probabilities[0][i]) for i in range(len(CLASSES))}
                        st.bar_chart(probs_dict)
                        
                    except Exception as e:
                        import traceback
                        st.error(f"Error during prediction: {str(e)}")
                        with st.expander("Show detailed error"):
                            st.code(traceback.format_exc())
                
    st.divider()
    st.caption("Disclaimer: Education Purpose only.")

if __name__ == "__main__":
    main()
