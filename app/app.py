import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from google import genai  # ✅ official Gemini SDK

# ==============================
# 1️⃣ Streamlit Config
# ==============================
st.set_page_config(page_title="🧠 Brain MRI Clinical Assistant", layout="wide")

st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #4facfe, #00f2fe);
border-radius: 15px; color: white;">
    <h1>🧠 Brain MRI AI Assistant for Doctors</h1>
    <h4>AI-powered MRI analysis, Grad-CAM visualization, and Gemini AI conversation</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# 2️⃣ Gemini API Setup
# ==============================
api_key = st.secrets["GEMINI_API_KEY"]  # replace with your real key
client = genai.Client(api_key=api_key)

# ==============================
# 3️⃣ Sidebar: Patient + Doctor Info
# ==============================
with st.sidebar:
    st.image("https://tse3.mm.bing.net/th/id/OIP.mkNQTA9e60kIima-KVR7PgHaFv?rs=1&pid=ImgDetMain&o=7&rm=3", width=80)
    st.title("🧾 Patient Intake")

    patient_name = st.text_input("Patient Name")
    doctor_choice = st.selectbox("Consulting Doctor", [
        "Dr. Ahmed (Radiologist)",
        "Dr. Sara (Neurosurgeon)",
        "Dr. Malik (Oncologist)",
        "Other"
    ])

    symptoms = st.multiselect("Select Symptoms", [
        "Persistent headache",
        "Seizures",
        "Vision problems",
        "Balance difficulty",
        "Nausea / vomiting",
        "Memory issues",
        "Weakness / numbness"
    ])

    st.info("This information will help Gemini understand condition severity.")


# ==============================
# 4️⃣ Class Names + Model Load
# ==============================
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(class_names))
    )
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model4.pth")
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        st.error(f"⚠️ Model loading error: {e}")
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==============================
# 5️⃣ Image Transformations
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 6️⃣ Grad-CAM Helper
# ==============================
def generate_gradcam(model, input_tensor, target_class=None):
    gradients, activations = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_full_backward_hook(save_gradient)

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    handle1.remove()
    handle2.remove()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * acts, axis=0), 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam

# ==============================
# 7️⃣ Upload & Predict
# ==============================
uploaded_file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🩻 Uploaded MRI", use_container_width=True)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    predicted_label = class_names[predicted_class.item()]
    st.markdown(f"### 🎯 **Predicted Tumor Type:** {predicted_label.title()}")
    st.markdown(f"**Confidence:** {confidence.item() * 100:.2f}%")

    # ---- Grad-CAM ----
    with st.spinner("🧩 Generating Grad-CAM..."):
        cam = generate_gradcam(model, input_tensor, predicted_class.item())
        img_np = np.array(image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="🔥 Tumor Focus", use_container_width=True)

    st.markdown("#### 📊 Class Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls.title()}: {probs[0][i].item() * 100:.2f}%")
        st.progress(float(probs[0][i].item()))

    # ==============================
    # 8️⃣ Instant Gemini Response
    # ==============================
    st.markdown("---")
    st.subheader("🤖 Summary of Your MRI Findings")

    symptom_list = ", ".join(symptoms) if symptoms else "No symptoms provided"

    intro_prompt = f"""
You are a careful, respectful medical assistant.
Do not say doctor viewed this report, just say "Based on the MRI analysis..."
and provide information in a patient-friendly manner and doctor will review later and you will get advice or 
treatment plan from doctor.
Patient Name: {patient_name} 
Doctor: {doctor_choice} 
Predicted Tumor Type: {predicted_label}
Confidence: {confidence.item() * 100:.2f}%
Symptoms Reported: {symptom_list}

Explain:
- Is this tumor type serious?
- When does it become critical?
- What symptoms should NOT be ignored?
- What questions should the patient ask their doctor?

Tone: patient-friendly, calm, reassuring, polite.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=intro_prompt
        )
        st.write(response.text.strip())
    except Exception as e:
        st.error(f"Gemini Error: {e}")

    # ==============================
    # 9️⃣ Continue Chat
    # ==============================
    st.markdown("---")
    st.subheader("💬 Ask More Questions")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_prompt = st.chat_input("Ask about treatment, progression, risk...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "text": user_prompt})
        context = "\n".join([f"{m['role']}: {m['text']}" for m in st.session_state.chat_history])

        follow_prompt = f"""
Continue conversation in a polite, patient-education style.

Here is conversation so far:
{context}
"""
        try:
            reply = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=follow_prompt
            )
            reply_text = reply.text.strip()
            st.session_state.chat_history.append({"role": "assistant", "text": reply_text})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "text": f"Error: {e}"})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            st.chat_message("assistant").write(msg["text"])

else:
    st.info("📎 Upload a Brain MRI image to start.")

# ==============================
# 9️⃣ Footer
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:14px;'>
Developed by <b>Nabeel Siddiqui</b> | EfficientNet-B0 + Grad-CAM + Gemini AI + Streamlit  
</div>
""", unsafe_allow_html=True)






