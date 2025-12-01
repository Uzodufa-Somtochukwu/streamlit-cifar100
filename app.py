import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os

MODEL_URL = "https://drive.google.com/uc?export=download&id=1aFo_wiE5fSKnb0Ny8dXULOi5kTI0MDdu"
MODEL_PATH = "model.pth"

# Download model if not already downloaded
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (92MB)... please wait. This happens only once."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cpu")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CIFAR-100 Classifier",
    page_icon="🖼️",
    layout="wide",
)

# ============================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================
st.markdown("""
    <style>
    .title-text {
        font-size: 38px;
        font-weight: 900;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-top: -10px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border: 1px solid #e6e9ef;
        margin-top: 20px;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown('<p class="title-text">🖼 CIFAR-100 Image Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload an image and watch the ResNet-50 model predict its class.</p>', unsafe_allow_html=True)
st.write("")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📌 Instructions")
st.sidebar.write("""
1. Upload any clear image (.jpg, .jpeg, .png).  
2. The model will preprocess it automatically.  
3. Get the predicted CIFAR-100 class + top-5 probabilities.  
""")

st.sidebar.write("👉 *Model: Custom ResNet-50 trained on CIFAR-100*")
st.sidebar.info("Made by Somto — for ML Deployment Assignment")

# ============================================================
# CIFAR-100 CLASS LIST
# ============================================================
classes = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","cactus","camel","can","castle","caterpillar",
    "cattle","chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup",
    "dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo",
    "keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle",
    "mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck",
    "pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road",
    "rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake",
    "spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger",
    "tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
]

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)

    checkpoint = torch.load("model.pth", map_location=DEVICE)
    

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ============================================================
# IMAGE TRANSFORM
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# MAIN UPLOAD SECTION
# ============================================================

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 🔄 Processing Image...")
        img = Image.open(uploaded_file).convert("RGB")

        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)
            pred_class = classes[top5_idx[0].item()]
            pred_conf = top5_prob[0].item() * 100

        # PREDICTION CARD
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown(f"## 🎯 Prediction: **{pred_class}**")
        st.markdown(f"### Confidence: **{pred_conf:.2f}%**")

        st.markdown("#### 🔝 Top 5 Predictions:")
        for i in range(5):
            st.write(f"**{classes[top5_idx[i]]}** — {top5_prob[i].item() * 100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.write("---")
st.caption("🚀 Built with Streamlit | ResNet-50 | CIFAR-100 — by Somto")
