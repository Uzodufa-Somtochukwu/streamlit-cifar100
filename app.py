import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os

# ============================================================
# MODEL DOWNLOAD CONFIG (HUGGINGFACE)
# ============================================================
MODEL_URL = "https://huggingface.co/Mhizdufa/resnet50-cifar100-somto/resolve/main/resnet50_cifar100.pth"
MODEL_PATH = "resnet50_cifar100.pth"

# Download model if needed
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (95MB)... please wait."):
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
# DARK MODE TOGGLE
# ============================================================
mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

background_color = "#ffffff" if not mode else "#0d1117"
text_color = "#000000" if not mode else "#f0f6fc"
subtitle_color = "#444444" if not mode else "#c9d1d9"
card_bg = "rgba(255,255,255,0.25)" if not mode else "rgba(13,17,23,0.45)"
card_border = "rgba(255,255,255,0.4)" if not mode else "rgba(240,246,252,0.2)"

# ============================================================
# GLASSMORPHIC UI
# ============================================================
st.markdown(f"""
    <style>
    body, .main {{
        background-color: {background_color} !important;
        color: {text_color} !important;
    }}

    .title-text {{
        font-size: 44px;
        font-weight: 900;
        text-align: center;
        color: #2563EB;
    }}

    .subtitle-text {{
        font-size: 20px;
        text-align: center;
        color: {subtitle_color};
        margin-bottom: 25px;
    }}

    .glass-card {{
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        background: {card_bg};
        border: 2px solid {card_border};
        padding: 28px;
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        margin-top: 20px;
    }}

    .pred-label {{
        font-size: 32px;
        font-weight: 800;
        color: #DC2626;
    }}

    .confidence {{
        font-size: 23px;
        font-weight: 700;
        color: #2563EB;
    }}

    .top5-title {{
        font-size: 19px;
        font-weight: 700;
        color: #2563EB;
        margin-top: 15px;
    }}

    .top5-item {{
        font-size: 17px;
        padding: 3px 0;
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown('<p class="title-text">CIFAR-100 Image Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload an image and let the ResNet-50 model predict its class.</p>', unsafe_allow_html=True)
st.write("")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📌 Instructions")
st.sidebar.write("""
1. Upload any image (.jpg, .jpeg, .png).  
2. The model will preprocess it.  
3. View the class prediction + top-5 probabilities.  
""")
st.sidebar.info("⚙ Model: Custom ResNet-50 trained on CIFAR-100\nMade by Somto ✨")

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
# LOAD MODEL (CACHED)
# ============================================================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ============================================================
# IMAGE TRANSFORMATIONS
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
        st.markdown('<p style="color:#2563EB;font-size:18px;font-weight:600;">🔍 Analyzing your image...</p>', unsafe_allow_html=True)

        img = Image.open(uploaded_file).convert("RGB")

        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)
            pred_class = classes[top5_idx[0].item()]
            pred_conf = top5_prob[0].item() * 100

        # ============================================================
        # GLASS CARD OUTPUT
        # ============================================================
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown(f'<p class="pred-label">🔮 Prediction: {pred_class}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">Confidence: {pred_conf:.2f}%</p>', unsafe_allow_html=True)

        st.markdown('<p class="top5-title">Top 5 Predictions:</p>', unsafe_allow_html=True)
        for i in range(5):
            st.markdown(
                f'<p class="top5-item">• {classes[top5_idx[i]]}: {top5_prob[i].item() * 100:.2f}%</p>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.write("---")
st.caption("✨ Built with Streamlit | ResNet-50 | CIFAR-100 — by Uzodufa Somto")
