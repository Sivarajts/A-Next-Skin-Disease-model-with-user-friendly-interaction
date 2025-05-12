import os

# Model Configuration
MODEL_PATH = "models/best_model.h5"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.5

# Disease Classes
DISEASE_CLASSES = [
    "Melanoma", "Benign Keratosis", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Vascular Lesion", "Dermatofibroma",
    "Squamous Cell Carcinoma"
]

# Disease Information
DISEASE_INFO = {
    "Melanoma": {
        "description": "A serious form of skin cancer that develops in melanocytes.",
        "symptoms": ["Asymmetrical moles", "Irregular borders", "Multiple colors", "Diameter > 6mm", "Evolving appearance"],
        "treatment": "Surgical removal, immunotherapy, targeted therapy",
        "prevention": "Regular skin checks, sun protection, avoiding tanning beds"
    },
    "Benign Keratosis": {
        "description": "Non-cancerous skin growths that appear as waxy, scaly patches.",
        "symptoms": ["Waxy appearance", "Scaly patches", "Light brown to black color"],
        "treatment": "Cryotherapy, curettage, topical treatments",
        "prevention": "Sun protection, regular skin checks"
    },
    "Basal Cell Carcinoma": {
        "description": "The most common type of skin cancer, usually appearing as a small, shiny bump.",
        "symptoms": ["Shiny bump", "Pink growth", "Sore that doesn't heal", "Scar-like area"],
        "treatment": "Surgical removal, radiation therapy, topical medications",
        "prevention": "Sun protection, regular skin checks"
    },
    "Actinic Keratosis": {
        "description": "Precancerous skin growths caused by sun damage.",
        "symptoms": ["Rough, scaly patches", "Pink or red color", "Itching or burning"],
        "treatment": "Cryotherapy, topical medications, photodynamic therapy",
        "prevention": "Sun protection, regular skin checks"
    },
    "Vascular Lesion": {
        "description": "Abnormal growth of blood vessels in the skin.",
        "symptoms": ["Red or purple patches", "Raised bumps", "Visible blood vessels"],
        "treatment": "Laser therapy, sclerotherapy, surgical removal",
        "prevention": "Protect from trauma, avoid blood thinners"
    },
    "Dermatofibroma": {
        "description": "A common benign skin tumor that appears as a hard, raised growth.",
        "symptoms": ["Hard, raised bump", "Brown or red color", "Slight itching"],
        "treatment": "Surgical removal if symptomatic",
        "prevention": "No specific prevention"
    },
    "Squamous Cell Carcinoma": {
        "description": "A type of skin cancer that develops in squamous cells.",
        "symptoms": ["Firm, red nodule", "Flat sore with scaly crust", "New sore on old scar"],
        "treatment": "Surgical removal, radiation therapy, chemotherapy",
        "prevention": "Sun protection, regular skin checks"
    }
}

# Expert Doctors List
EXPERT_DOCTORS = [
    {"name": "Dr. Ravi Kumar", "hospital": "Apollo Hospital, Chennai", "specialization": "Dermatology"},
    {"name": "Dr. Priya Sharma", "hospital": "Fortis Hospital, Coimbatore", "specialization": "Skin Cancer Specialist"},
    {"name": "Dr. Karthik Reddy", "hospital": "MIOT Hospital, Chennai", "specialization": "Cosmetic Dermatology"},
    {"name": "Dr. Anitha Raj", "hospital": "PSG Hospital, Coimbatore", "specialization": "Pediatric Dermatology"},
    {"name": "Dr. Suresh Babu", "hospital": "SRM Hospital, Trichy", "specialization": "General Dermatology"}
]

# Database Configuration
DB_PATH = "appointments.db"

# Email Configuration
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 465,
    "use_ssl": True
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 150
} 