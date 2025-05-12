import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from PIL import Image
import json
import requests
from openai import OpenAI

# Set page config
st.set_page_config(
    page_title="Skin Disease Detection System",
    page_icon="🏥",
    layout="wide"
)

# Expert Doctors List with specialization
EXPERT_DOCTORS = [
    {"name": "Dr. Ravi Kumar", "hospital": "Apollo Hospital, Chennai", "specialization": "Dermatology"},
    {"name": "Dr. Priya Sharma", "hospital": "Fortis Hospital, Coimbatore", "specialization": "Skin Cancer Specialist"},
    {"name": "Dr. Karthik Reddy", "hospital": "MIOT Hospital, Chennai", "specialization": "Cosmetic Dermatology"},
    {"name": "Dr. Anitha Raj", "hospital": "PSG Hospital, Coimbatore", "specialization": "Pediatric Dermatology"},
    {"name": "Dr. Suresh Babu", "hospital": "SRM Hospital, Trichy", "specialization": "General Dermatology"}
]

# Disease Information
DISEASE_INFO = {
    "Melanoma": {
        "description": "A serious form of skin cancer that develops in melanocytes.",
        "common_name": "Skin Cancer (Melanoma)",
        "symptoms": ["Asymmetrical moles", "Irregular borders", "Multiple colors", "Diameter > 6mm", "Evolving appearance"],
        "treatment": "Surgical removal, immunotherapy, targeted therapy",
        "prevention": "Regular skin checks, sun protection, avoiding tanning beds"
    },
    "Benign Keratosis": {
        "description": "Non-cancerous skin growths that appear as waxy, scaly patches.",
        "common_name": "Benign Skin Growth",
        "symptoms": ["Waxy appearance", "Scaly patches", "Light brown to black color"],
        "treatment": "Cryotherapy, curettage, topical treatments",
        "prevention": "Sun protection, regular skin checks"
    },
    "Basal Cell Carcinoma": {
        "description": "The most common type of skin cancer, usually appearing on sun-exposed areas.",
        "common_name": "Skin Cancer (Basal Cell)",
        "symptoms": ["Pearly or waxy bump", "Flat, flesh-colored lesion", "Brown scar-like lesion"],
        "treatment": "Surgical removal, radiation therapy, topical medications",
        "prevention": "Sun protection, regular skin checks"
    },
    "Actinic Keratosis": {
        "description": "Rough, scaly patches on sun-damaged skin that can develop into skin cancer.",
        "common_name": "Precancerous Skin Lesion",
        "symptoms": ["Rough, dry patches", "Pink, red, or brown color", "Itching or burning"],
        "treatment": "Cryotherapy, topical medications, photodynamic therapy",
        "prevention": "Sun protection, regular skin checks"
    },
    "Vascular Lesion": {
        "description": "Abnormal growth of blood vessels in the skin.",
        "common_name": "Blood Vessel Lesion",
        "symptoms": ["Red or purple patches", "Raised or flat lesions", "May be present at birth"],
        "treatment": "Laser therapy, surgery, medication",
        "prevention": "Protection from trauma, sun protection"
    },
    "Dermatofibroma": {
        "description": "Common benign skin tumor that usually appears on the legs.",
        "common_name": "Benign Skin Nodule",
        "symptoms": ["Hard, raised growth", "Brown or purple color", "May be itchy"],
        "treatment": "Surgical removal if symptomatic",
        "prevention": "Protection from trauma"
    },
    "Squamous Cell Carcinoma": {
        "description": "Common type of skin cancer that can spread to other parts of the body.",
        "common_name": "Skin Cancer (Squamous Cell)",
        "symptoms": ["Firm, red nodule", "Flat lesion with scaly surface", "Sore that doesn't heal"],
        "treatment": "Surgical removal, radiation therapy, chemotherapy",
        "prevention": "Sun protection, regular skin checks"
    },
    "Acne": {
        "description": "A common skin condition causing pimples, blackheads, and cysts.",
        "common_name": "Pimples/Acne",
        "symptoms": ["Red pimples", "Whiteheads", "Blackheads", "Cysts", "Oily skin"],
        "treatment": "Topical creams, oral medications, proper skin hygiene",
        "prevention": "Regular face washing, avoid oily products, healthy diet"
    },
    "Scars": {
        "description": "Areas of fibrous tissue that replace normal skin after injury.",
        "common_name": "Skin Scars",
        "symptoms": ["Discolored patches", "Raised or sunken areas", "Irregular texture"],
        "treatment": "Laser therapy, creams, surgical revision",
        "prevention": "Proper wound care, avoid picking at skin"
    },
    "Clear Skin": {
        "description": "No visible skin disease or abnormality detected.",
        "common_name": "Healthy Skin",
        "symptoms": ["Even tone", "No lesions", "Smooth texture"],
        "treatment": "No treatment needed",
        "prevention": "Maintain good hygiene, sun protection, healthy lifestyle"
    },
    "Not Skin": {
        "description": "The uploaded image does not appear to be a skin image.",
        "common_name": "Not a Skin Image",
        "symptoms": [],
        "treatment": "Please upload a clear image of skin.",
        "prevention": "Ensure the image is of the affected skin area."
    }
}

# Initialize session state for storing data
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'appointments' not in st.session_state:
    st.session_state.appointments = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'doctors' not in st.session_state:
    st.session_state.doctors = EXPERT_DOCTORS
if 'disease_info' not in st.session_state:
    st.session_state.disease_info = DISEASE_INFO

# Constants
MODEL_PATH = "models/best_model.h5"
DISEASE_CLASSES = [
    "Melanoma", "Benign Keratosis", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Vascular Lesion", "Dermatofibroma",
    "Squamous Cell Carcinoma"
]
CONFIDENCE_THRESHOLD = 0.5

# Chatbot suggestions
CHAT_SUGGESTIONS = [
    "What are the symptoms of melanoma?",
    "How can I prevent skin cancer?",
    "What are common skin diseases?",
    "How to treat acne?",
    "What causes eczema?",
    "How to protect skin from sun damage?",
    "What are the early signs of skin cancer?",
    "How to maintain healthy skin?",
    "What are the different types of skin rashes?",
    "How to treat psoriasis?"
]

# Initialize OpenAI client with error handling
try:
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    client = None

# Initialize model as a global variable
model = None

# Load the trained model with custom objects
@st.cache_resource
def load_model():
    global model
    try:
        # Create a new model with the same architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(DISEASE_CLASSES), activation='softmax')
        ])
        
        # Load weights if available
        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model at startup
model = load_model()

# Email function with better error handling
def send_confirmation_email(to_email, name, doctor, date, time, disease):
    try:
        sender_email = st.secrets["email"]["sender"]
        sender_password = st.secrets["email"]["password"]
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = "Appointment Confirmation - Skin Disease Consultation"
        
        body = f"""
        Dear {name},
        
        Your appointment has been confirmed:
        Doctor: {doctor}
        Date: {date}
        Time: {time}
        Concern: {disease}
        
        Please arrive 15 minutes before your scheduled time.
        If you need to reschedule, please contact us at least 24 hours in advance.
        
        Best regards,
        Skin Disease Detection Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

# Prediction function with better error handling
def predict_disease(image):
    try:
        if model is None:
            st.error("Model not loaded properly. Please try again later.")
            return None, 0, {}

        # Heuristic: Check if image is likely a skin image (RGB, not grayscale, not too small, skin-like color)
        if len(image.shape) != 3 or image.shape[2] != 3 or min(image.shape[0], image.shape[1]) < 50:
            return "Not Skin", 0, DISEASE_INFO["Not Skin"]
        avg_color = np.mean(image, axis=(0, 1))
        # Simple check: skin color range (very rough, for demo)
        if not (80 < avg_color[0] < 220 and 50 < avg_color[1] < 200 and 40 < avg_color[2] < 180):
            return "Not Skin", 0, DISEASE_INFO["Not Skin"]

        # Preprocess image
        img = cv2.resize(image, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        predictions = model.predict(img)
        predicted_class = DISEASE_CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]) * 100)

        # Heuristic: If confidence is very low, say clear skin
        if confidence < 20:
            return "Clear Skin", confidence, DISEASE_INFO["Clear Skin"]

        # Simple acne/scar detection (very basic, for demo)
        # If many small red spots, call it acne
        hsv = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_RGB2HSV)
        mask_acne = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        acne_pixels = np.sum(mask_acne > 0)
        if acne_pixels > 500:  # threshold for demo
            return "Acne", 80, DISEASE_INFO["Acne"]
        # If many irregular patches, call it scars
        gray = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        scar_pixels = np.sum(edges > 0)
        if scar_pixels > 3000:
            return "Scars", 70, DISEASE_INFO["Scars"]

        # Save prediction to history
        st.session_state.predictions.append({
            "disease": predicted_class,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        return predicted_class, confidence, DISEASE_INFO.get(predicted_class, {})
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, 0, {}

# Local chatbot function with suggestions
def get_local_response(user_message, prediction_context=None):
    # Convert user message to lowercase for easier matching
    message = user_message.lower()
    
    # Update suggestions based on user input
    if len(message) > 2:
        filtered_suggestions = [s for s in CHAT_SUGGESTIONS if message.lower() in s.lower()]
        st.session_state.suggestions = filtered_suggestions[:5]  # Show top 5 suggestions
    
    # Check for common questions about skin diseases
    for disease, info in DISEASE_INFO.items():
        if disease.lower() in message:
            return f"{disease}:\n\nDescription: {info['description']}\n\nSymptoms: {', '.join(info['symptoms'])}\n\nTreatment: {info['treatment']}\n\nPrevention: {info['prevention']}"
    
    # Check for specific keywords
    if any(word in message for word in ["symptom", "sign", "indication"]):
        return "Common skin disease symptoms include:\n- Changes in skin color or texture\n- Itching or burning sensations\n- Pain or tenderness\n- Unusual growths or lesions\n- Changes in existing moles\n\nIf you notice any of these symptoms, please consult a healthcare professional."
    
    if any(word in message for word in ["prevent", "avoid", "protection"]):
        return "To prevent skin diseases:\n1. Use sunscreen daily\n2. Avoid excessive sun exposure\n3. Perform regular skin self-examinations\n4. Maintain good hygiene\n5. Keep skin moisturized\n6. Avoid sharing personal items\n7. Get regular check-ups"
    
    if any(word in message for word in ["treatment", "cure", "therapy"]):
        return "Common skin disease treatments include:\n1. Topical medications\n2. Oral medications\n3. Light therapy\n4. Surgical procedures\n5. Cryotherapy\n6. Laser treatment\n\nThe specific treatment depends on the condition. Please consult a healthcare professional for proper diagnosis and treatment."
    
    # General responses
    if "hello" in message or "hi" in message:
        return "Hello! I'm your skin disease assistant. How can I help you today?"
    elif "help" in message:
        return "I can help you with:\n1. Information about skin diseases\n2. Symptoms and treatments\n3. Prevention tips\n4. Booking appointments\nJust ask me about any skin condition!"
    elif "thank" in message:
        return "You're welcome! Let me know if you have any other questions."
    elif "bye" in message:
        return "Goodbye! Take care of your skin!"
    else:
        return "I'm not sure about that. Could you please rephrase your question or ask about a specific skin condition?"

# Chatbot function with fallback to local model
def chat_with_gpt(user_message, prediction_context=None):
    try:
        # First try OpenAI
        if client:
            messages = []
            if prediction_context:
                messages.append({
                    "role": "system",
                    "content": f"Previous prediction context: {prediction_context}"
                })
            
            messages.append({"role": "user", "content": user_message})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            bot_response = response.choices[0].message.content
        else:
            # Fallback to local model
            bot_response = get_local_response(user_message, prediction_context)
        
        # Save chat history
        st.session_state.chat_history.append({
            "user_message": user_message,
            "bot_response": bot_response,
            "prediction_context": prediction_context,
            "timestamp": datetime.now().isoformat()
        })
        
        return bot_response
    except Exception as e:
        st.error(f"Chatbot error: {str(e)}")
        return get_local_response(user_message, prediction_context)

# Main UI
st.title("🏥 Skin Disease Detection & Consultation System")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Disease Detection", "Book Appointment", "Chatbot", "History"])

# Disease Detection Page
if page == "Disease Detection":
    st.header("📷 Skin Disease Detection")
    
    uploaded_file = st.file_uploader("Upload an image of the affected area", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to numpy array for processing
        image_np = np.array(image)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                disease, confidence, info = predict_disease(image_np)
                
                if disease:
                    # Show both medical and common name
                    common_name = info.get("common_name", "")
                    if disease == "Not Skin":
                        st.error(f"❌ {info['description']}")
                    elif disease == "Clear Skin":
                        st.success(f"✅ Detected: Clear Skin (Healthy Skin)")
                        st.info(f"🔍 Confidence: {confidence:.2f}%")
                    elif disease == "Acne":
                        st.success(f"✅ Detected: Acne (Pimples)")
                        st.info(f"🔍 Confidence: {confidence:.2f}%")
                    elif disease == "Scars":
                        st.success(f"✅ Detected: Scars")
                        st.info(f"🔍 Confidence: {confidence:.2f}%")
                    else:
                        st.success(f"✅ Predicted Disease: {disease} (Common: {common_name})")
                        st.info(f"🔍 Confidence: {confidence:.2f}%")
                    
                    # Display detailed information
                    st.subheader("📋 Disease Information")
                    st.write(info.get("description", ""))
                    
                    st.subheader("⚠️ Common Symptoms")
                    for symptom in info.get("symptoms", []):
                        st.write(f"• {symptom}")
                    
                    st.subheader("💊 Treatment Options")
                    st.write(info.get("treatment", ""))
                    
                    st.subheader("🛡️ Prevention Tips")
                    st.write(info.get("prevention", ""))
                    
                    # Book appointment button
                    if disease not in ["Not Skin", "Clear Skin"]:
                        if st.button("Book Appointment with Specialist"):
                            st.session_state['predicted_disease'] = disease
                            st.session_state['confidence'] = confidence
                            st.experimental_rerun()

# Appointment Booking Page
elif page == "Book Appointment":
    st.header("📅 Book an Appointment")
    
    # Pre-fill disease if coming from prediction
    predicted_disease = st.session_state.get('predicted_disease', '')
    
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone Number")
    disease = st.text_input("Concern/Diagnosis", value=predicted_disease)
    
    # Doctor selection with specialization
    doctor_options = [f"{doc['name']} - {doc['specialization']} ({doc['hospital']})" 
                     for doc in EXPERT_DOCTORS]
    selected_doctor = st.selectbox("Select Doctor", doctor_options)
    
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Preferred Date")
    with col2:
        time = st.time_input("Preferred Time")
    
    if st.button("Confirm Appointment"):
        if name and email and phone and disease and selected_doctor and date and time:
            try:
                # Save appointment
                st.session_state.appointments.append({
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "date": date.strftime('%Y-%m-%d'),
                    "time": time.strftime('%H:%M'),
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                })
                
                # Send confirmation email
                if send_confirmation_email(email, name, selected_doctor, 
                                        date.strftime('%Y-%m-%d'), 
                                        time.strftime('%H:%M'), disease):
                    st.success("✅ Appointment confirmed! A confirmation email has been sent.")
                else:
                    st.warning("⚠️ Appointment saved but email could not be sent.")
            except Exception as e:
                st.error(f"Error booking appointment: {str(e)}")
        else:
            st.error("⚠️ Please fill all fields correctly.")

# Chatbot Page
elif page == "Chatbot":
    st.header("🤖 AI Dermatology Assistant")
    
    # Get prediction context if available
    prediction_context = None
    if 'predicted_disease' in st.session_state:
        prediction_context = {
            'disease': st.session_state.predicted_disease,
            'confidence': st.session_state.confidence
        }
    
    # Chat input with suggestions
    user_input = st.text_input("Ask about skin diseases, symptoms, or treatments:")
    
    # Show suggestions if available
    if st.session_state.suggestions:
        st.write("💡 Suggested questions:")
        for suggestion in st.session_state.suggestions:
            if st.button(suggestion, key=suggestion):
                user_input = suggestion
                st.experimental_rerun()
    
    if user_input:
        with st.spinner("Thinking..."):
            response = chat_with_gpt(user_input, prediction_context)
            st.session_state.chat_history.append({
                "user_message": user_input,
                "bot_response": response,
                "timestamp": datetime.now().isoformat()
            })
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"👤 **You:** {chat['user_message']}")
        st.write(f"🤖 **Bot:** {chat['bot_response']}")
        st.write("---")

# History Page
elif page == "History":
    st.header("📊 Your History")
    
    tab1, tab2, tab3 = st.tabs(["Prediction History", "Appointment History", "Chat History"])
    
    with tab1:
        st.subheader("Recent Predictions")
        if st.session_state.predictions:
            for pred in st.session_state.predictions:
                st.write(f"**Disease:** {pred['disease']}")
                st.write(f"**Confidence:** {pred['confidence']:.2f}%")
                st.write(f"**Date:** {pred['timestamp']}")
                st.write("---")
        else:
            st.info("No prediction history available.")
    
    with tab2:
        st.subheader("Appointment History")
        if st.session_state.appointments:
            for apt in st.session_state.appointments:
                with st.expander(f"Appointment on {apt['date']} at {apt['time']}"):
                    st.write(f"**Patient:** {apt['name']}")
                    st.write(f"**Email:** {apt['email']}")
                    st.write(f"**Phone:** {apt['phone']}")
                    st.write(f"**Date:** {apt['date']}")
                    st.write(f"**Time:** {apt['time']}")
                    st.write(f"**Status:** {apt['status']}")
        else:
            st.info("No appointment history available.")
    
    with tab3:
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write(f"**You:** {chat['user_message']}")
                st.write(f"**Bot:** {chat['bot_response']}")
                st.write(f"**Time:** {chat['timestamp']}")
                st.write("---")
        else:
            st.info("No chat history available.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ for better skin health</p>
    <p>Note: This system is for preliminary screening only. Always consult a healthcare professional for proper diagnosis.</p>
</div>
""", unsafe_allow_html=True)
