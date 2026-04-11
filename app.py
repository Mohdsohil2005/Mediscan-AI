from flask import Flask, request, render_template, redirect, url_for, send_from_directory, make_response
import cv2
import numpy as np
import os
import pickle
import re
from io import BytesIO
from pathlib import Path
from textwrap import wrap
# from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from uuid import uuid4
from PIL import Image, ImageDraw, ImageFont
from ml_utils import build_tta_batch, extract_focus_roi, normalize_prediction_label, summarize_prediction

import os

# if not os.path.exists("models/CNN_Covid19_Xray_Version.h5"):
#     import gdown
#     url = "YOUR_GOOGLE_DRIVE_LINK"
#     output = "models/CNN_Covid19_Xray_Version.h5"
#     gdown.download(url, output, quiet=False)


try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

model_gemini = genai.GenerativeModel("gemini-1.5-flash") if genai is not None and GEMINI_API_KEY else None

def is_hindi_text(text):
    hindi_words = {
        "namaste", "hello ji", "hey ji", "hy", "haan", "nahi", "kya", "kaise", "bukhar",
        "khansi", "dawai", "doctor", "sehat", "bimari", "dard", "saans", "gala", "sardi"
    }
    lowered = text.lower()
    if re.search(r"[\u0900-\u097F]", text):
        return True
    return any(word in lowered for word in hindi_words)

def is_greeting(text):
    normalized = re.sub(r"[^a-zA-Z ]", " ", text.lower()).strip()
    greetings = {
        "hello", "hey", "hi", "hy", "hii", "helo", "namaste", "namaskar", "hello ji", "hi ji", "hey ji"
    }
    return normalized in greetings

def is_medical_question(query):
    medical_keywords = {
        "covid", "fever", "pain", "infection", "disease", "symptom", "symptoms",
        "treatment", "medicine", "doctor", "virus", "lungs", "lung", "xray", "x-ray",
        "cough", "cold", "flu", "pneumonia", "breathing", "breath", "chest", "hospital",
        "scan", "diagnosis", "medical", "tablet", "paracetamol", "headache", "bodyache",
        "sore", "throat", "asthma", "oxygen", "bp", "sugar", "diabetes", "infection",
        "vomiting", "nausea", "diarrhea", "injury", "fracture", "health", "illness"
    }
    query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
    return bool(query_words & medical_keywords)

def local_medical_response(user_input):
    query = user_input.lower()
    hindi = is_hindi_text(user_input)

    if is_greeting(user_input):
        if hindi:
            return "Namaste, main aapki medical questions me madad kar sakta hoon. Aap apna health ya medical sawal pooch sakte hain."
        return "Hello, I can help with medical and health-related questions. You can ask your medical question."

    if any(word in query for word in ["covid", "    "]):
        if hindi:
            return (
                "COVID se jude aam lakshan bukhar, khansi, gala dard, thakan aur saans lene me dikkat ho sakte hain. "
                "Agar saans ki takleef zyada ho ya symptoms severe hon to turant doctor se sampark karein. "
                "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."
            )
        return (
            "Common COVID-related symptoms can include fever, cough, sore throat, tiredness, and breathing difficulty. "
            "If symptoms are severe, especially shortness of breath, seek medical care quickly. "
            "This is not a medical diagnosis. Please consult a doctor."
        )

    if any(word in query for word in ["xray", "x-ray", "scan", "chest"]):
        if hindi:
            return (
                "Chest X-ray ko sahi tarah samajhne ke liye qualified doctor ya radiologist ki salah zaroori hoti hai. "
                "Agar aapko chest pain, bukhar, khansi, ya saans lene me dikkat hai to doctor se consult karein. "
                "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."
            )
        return (
            "Chest X-ray reports should be reviewed by a qualified doctor or radiologist because image findings need medical interpretation. "
            "If you have chest pain, fever, cough, or breathing trouble, please get professional medical advice. "
            "This is not a medical diagnosis. Please consult a doctor."
        )

    if any(word in query for word in ["fever","bukhar", "cold", "cough", "flu", "sore", "throat"]):
        if hindi:
            return (
                "Halka bukhar, khansi ya sardi me aaram, paani aur basic care madad kar sakti hai. "
                "Lekin agar bukhar zyada ho, lamba chale, ya saans ki dikkat, chest pain, ya zyada kamzori ho to doctor ko dikhayein. "
                "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."
            )
        return (
            "For mild fever, cough, or cold, rest, fluids, and basic supportive care are often helpful. "
            "If symptoms are high, prolonged, or include breathing trouble, chest pain, or weakness, visit a doctor. "
            "This is not a medical diagnosis. Please consult a doctor."
        )

    if any(word in query for word in ["pneumonia", "breathing", "breath", "oxygen", "lungs", "lung"]):
        if hindi:
            return (
                "Saans ki dikkat ya pneumonia jaise lakshan ko seriously lena chahiye, khas kar jab fast breathing, low oxygen, chest pain, ya confusion ho. "
                "Aise symptoms me jaldi medical evaluation karwana zaroori hai. "
                "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."
            )
        return (
            "Breathing problems or possible pneumonia should be taken seriously, especially if there is fast breathing, low oxygen, chest pain, or confusion. "
            "Please seek urgent medical evaluation if symptoms are significant. "
            "This is not a medical diagnosis. Please consult a doctor."
        )

    if any(word in query for word in ["medicine", "tablet", "paracetamol", "treatment"]):
        if hindi:
            return (
                "Dawai sahi bimari, age, dose aur medical history ke hisab se leni chahiye. "
                "Serious symptoms me bina doctor ki salah ke self-medication se bachein aur doctor ya pharmacist se sahi advice lein. "
                "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."
            )
        return (
            "Medicines should be taken according to the correct condition, age, dose, and medical history. "
            "Avoid self-medicating for serious symptoms and speak to a doctor or pharmacist for proper treatment advice. "
            "This is not a medical diagnosis. Please consult a doctor."
        )

    if hindi:
        return (
            "Main sirf medical ya health-related sawalon me madad kar sakta hoon. "
            "Aap symptoms, illness, treatment basics, COVID, pneumonia, chest X-ray ya health topics ke baare me pooch sakte hain. "
            "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."
        )
    return (
        "I can help only with medical or health-related questions. Please ask about symptoms, illness, treatment basics, COVID, pneumonia, chest X-ray, or related topics. "
        "This is not a medical diagnosis. Please consult a doctor."
    )


def medical_chatbot(user_input):
    cleaned_input = user_input.strip()
    hindi = is_hindi_text(cleaned_input)

    if is_greeting(cleaned_input):
        return local_medical_response(cleaned_input)

    if not is_medical_question(cleaned_input):
        if hindi:
            return "Sorry, main sirf medical-related questions ka answer de sakta hoon."
        return "Sorry, I can answer only medical-related questions."

    if model_gemini is None:
        return local_medical_response(cleaned_input)

    prompt = f"""
You are a medical assistant chatbot for a chest X-ray website.
- Answer ONLY medical or health-related questions.
- If the question is non-medical, reply exactly: "Sorry, I can answer only medical-related questions."
- Keep the answer short, simple, and safe.
- Never claim to give a confirmed diagnosis.
- If the user writes in Hindi, reply fully in Hindi.
- If the user writes in English, reply in English.
- End every valid English medical answer with: "This is not a medical diagnosis. Please consult a doctor."
- End every valid Hindi medical answer with: "Yeh medical diagnosis nahi hai. Kripya doctor se salah lein."

User question: {cleaned_input}
"""

    try:
        response = model_gemini.generate_content(prompt)
        reply = (response.text or "").strip()
        return reply or local_medical_response(cleaned_input)
    except Exception:
        return local_medical_response(cleaned_input)


def render_page(template_name, active_page):
    return render_template(template_name, active_page=active_page)

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and label encoder
# model = load_model('./models/CNN_Covid19_Xray_Version.h5')  # Replace with your model path
le = pickle.load(open("./models/Label_encoder.pkl", 'rb'))  # Load the label encoder

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
PDF_FONT_PATH = Path(r"C:\Windows\Fonts\Nirmala.ttf")
PDF_FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\NirmalaB.ttf")


def allowed_file_extension(filename):
    return Path(filename or "").suffix.lower() in ALLOWED_EXTENSIONS


def remove_file_safely(file_path):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass

def render_invalid_result(filename=None, message='Invalid image. Please upload or capture a chest X-ray image only.'):
    return render_template(
        'result.html',
        filename=filename,
        error_message=message
    )


def build_report_context(filename, predicted_label, confidence_score):
    label = normalize_prediction_label(predicted_label)
    confidence_percent = round(float(confidence_score) * 100, 2)

    common = {
        "general_care": [
            "Drink enough water and stay well hydrated.",
            "Take proper rest and avoid overexertion.",
            "Keep meals light, clean, and easy to digest.",
            "Do not start antibiotics or steroids without a doctor's advice.",
        ],
        "general_care_hi": [
            "पर्याप्त पानी पिएं और शरीर को हाइड्रेटेड रखें।",
            "पूरा आराम करें और ज्यादा मेहनत से बचें।",
            "हल्का, साफ और आसानी से पचने वाला भोजन लें।",
            "बिना डॉक्टर की सलाह के एंटीबायोटिक या स्टेरॉइड शुरू न करें।",
        ],
        "warning_signs": [
            "Breathing difficulty or fast breathing",
            "Persistent chest pain",
            "High fever that does not improve",
            "Low oxygen or bluish lips",
            "Confusion, severe weakness, or fainting",
        ],
        "warning_signs_hi": [
            "सांस लेने में दिक्कत या बहुत तेज सांस चलना",
            "लगातार छाती में दर्द",
            "तेज बुखार जो कम न हो",
            "ऑक्सीजन कम होना या होंठ नीले पड़ना",
            "घबराहट, बहुत कमजोरी या बेहोशी जैसा महसूस होना",
        ],
        "disclaimer": "This report is AI-assisted and not a final medical diagnosis. Please consult a qualified doctor for confirmation.",
        "disclaimer_hi": "यह रिपोर्ट एआई आधारित सहायता है, अंतिम मेडिकल डायग्नोसिस नहीं। कृपया पुष्टि के लिए योग्य डॉक्टर से सलाह लें।",
    }

    report_by_label = {
        "COVID": {
            "display_label": "COVID",
            "display_label_hi": "कोविड",
            "risk_level": "High Risk",
            "risk_level_hi": "उच्च जोखिम",
            "summary": "The uploaded chest X-ray appears more consistent with COVID-related findings. Medical confirmation is important.",
            "summary_hi": "अपलोड की गई चेस्ट एक्स-रे इमेज कोविड से जुड़े संकेतों के अधिक करीब दिख रही है। मेडिकल पुष्टि जरूरी है।",
            "next_steps": [
                "Consult a doctor or chest specialist as early as possible.",
                "Monitor oxygen level if you have cough, fever, or breathing trouble.",
                "Limit close contact with others if symptoms are present.",
            ],
            "next_steps_hi": [
                "जितनी जल्दी हो सके डॉक्टर या चेस्ट स्पेशलिस्ट से सलाह लें।",
                "अगर खांसी, बुखार या सांस की दिक्कत है तो ऑक्सीजन लेवल मॉनिटर करें।",
                "लक्षण हों तो दूसरों से बहुत नजदीकी संपर्क कम रखें।",
            ],
            "eat_more": [
                "Warm fluids, soups, dal, khichdi, fruits rich in vitamin C",
                "Protein-rich foods like eggs, paneer, curd, pulses if tolerated",
                "Soft homemade meals that are easy to digest",
            ],
            "eat_more_hi": [
                "गुनगुने तरल पदार्थ, सूप, दाल, खिचड़ी और विटामिन C वाले फल लें।",
                "अंडे, पनीर, दही, दालें जैसे प्रोटीन युक्त भोजन लें यदि शरीर सहन करे।",
                "घर का हल्का और आसानी से पचने वाला भोजन लें।",
            ],
            "avoid": [
                "Junk food, cold drinks, smoking, and alcohol",
                "Heavy oily food if you have weakness or fever",
                "Self-medication without medical guidance",
            ],
            "avoid_hi": [
                "जंक फूड, ठंडे ड्रिंक्स, धूम्रपान और शराब से बचें।",
                "कमजोरी या बुखार होने पर बहुत तला-भुना और भारी भोजन न लें।",
                "बिना मेडिकल सलाह के खुद से दवा न लें।",
            ],
            "doctor_when": [
                "If oxygen drops, chest pain appears, or breathing becomes difficult",
                "If fever is high for several days",
                "If you are elderly, pregnant, or have diabetes/asthma/heart disease",
            ],
            "doctor_when_hi": [
                "अगर ऑक्सीजन कम हो, छाती में दर्द हो या सांस लेना कठिन हो जाए।",
                "अगर कई दिनों तक तेज बुखार बना रहे।",
                "अगर उम्र ज्यादा है, गर्भावस्था है, या डायबिटीज/अस्थमा/दिल की बीमारी है।",
            ],
        },
        "NORMAL": {
            "display_label": "NORMAL",
            "display_label_hi": "सामान्य",
            "risk_level": "Low Risk",
            "risk_level_hi": "कम जोखिम",
            "summary": "The uploaded chest X-ray appears closer to the normal class. Even then, symptoms should still be reviewed clinically.",
            "summary_hi": "अपलोड की गई चेस्ट एक्स-रे इमेज सामान्य वर्ग के अधिक करीब दिख रही है। फिर भी लक्षणों की डॉक्टर से जांच जरूरी हो सकती है।",
            "next_steps": [
                "If you still have symptoms, discuss them with a doctor.",
                "Keep observing fever, cough, or breathing changes.",
                "Continue healthy food, hydration, and rest.",
            ],
            "next_steps_hi": [
                "अगर फिर भी लक्षण हैं तो डॉक्टर से चर्चा करें।",
                "बुखार, खांसी या सांस में बदलाव पर नजर रखें।",
                "स्वस्थ भोजन, पानी और पर्याप्त आराम जारी रखें।",
            ],
            "eat_more": [
                "Normal balanced home-cooked meals",
                "Fresh fruits, vegetables, curd, pulses, and enough water",
                "Light protein-rich foods for recovery and immunity",
            ],
            "eat_more_hi": [
                "संतुलित घर का बना सामान्य भोजन लें।",
                "ताजे फल, सब्जियां, दही, दालें और पर्याप्त पानी लें।",
                "रिकवरी और इम्यूनिटी के लिए हल्का प्रोटीन युक्त भोजन लें।",
            ],
            "avoid": [
                "Skipping meals, dehydration, or excessive junk food",
                "Ignoring symptoms just because the screen looks normal",
                "Smoking and poor sleep routine",
            ],
            "avoid_hi": [
                "भोजन छोड़ना, पानी कम पीना या बहुत ज्यादा जंक फूड खाना।",
                "सिर्फ रिपोर्ट सामान्य दिखने पर लक्षणों को नजरअंदाज करना।",
                "धूम्रपान और खराब नींद की आदतें।",
            ],
            "doctor_when": [
                "If cough, fever, or chest pain continues",
                "If breathing trouble develops later",
                "If symptoms worsen despite a normal-looking report",
            ],
            "doctor_when_hi": [
                "अगर खांसी, बुखार या छाती दर्द बना रहे।",
                "अगर बाद में सांस की दिक्कत शुरू हो जाए।",
                "अगर सामान्य रिपोर्ट के बाद भी लक्षण बढ़ते जाएं।",
            ],
        },
        "PNEUMONIA": {
            "display_label": "VIRAL PNEUMONIA",
            "display_label_hi": "वायरल निमोनिया",
            "risk_level": "Needs Attention",
            "risk_level_hi": "ध्यान आवश्यक",
            "summary": "The uploaded chest X-ray appears more consistent with pneumonia-type findings. A doctor should review it soon.",
            "summary_hi": "अपलोड की गई चेस्ट एक्स-रे इमेज निमोनिया जैसे संकेतों के अधिक करीब दिख रही है। डॉक्टर से जल्दी समीक्षा करवानी चाहिए।",
            "next_steps": [
                "Book a doctor visit for proper chest evaluation.",
                "Monitor fever, cough, mucus, and breathing difficulty.",
                "Rest well and avoid dust, smoke, and exertion.",
            ],
            "next_steps_hi": [
                "चेस्ट की सही जांच के लिए डॉक्टर का अपॉइंटमेंट लें।",
                "बुखार, खांसी, बलगम और सांस की दिक्कत पर नजर रखें।",
                "अच्छा आराम करें और धूल, धुएं तथा ज्यादा मेहनत से बचें।",
            ],
            "eat_more": [
                "Warm liquids, soups, coconut water, and light nutritious meals",
                "Protein-rich foods, fruits, and soft cooked vegetables",
                "Hydration-focused meals to support recovery",
            ],
            "eat_more_hi": [
                "गुनगुने तरल पदार्थ, सूप, नारियल पानी और हल्का पौष्टिक भोजन लें।",
                "प्रोटीन युक्त भोजन, फल और नरम पकी हुई सब्जियां लें।",
                "रिकवरी के लिए शरीर को हाइड्रेट रखने वाला भोजन लें।",
            ],
            "avoid": [
                "Smoking, polluted air, very cold drinks, and heavy fried food",
                "Ignoring persistent fever or breathlessness",
                "Taking random medicines without doctor's advice",
            ],
            "avoid_hi": [
                "धूम्रपान, प्रदूषित हवा, बहुत ठंडे पेय और भारी तला भोजन से बचें।",
                "लगातार बुखार या सांस फूलने को नजरअंदाज न करें।",
                "बिना डॉक्टर की सलाह के कोई भी दवा न लें।",
            ],
            "doctor_when": [
                "If cough and fever continue or worsen",
                "If breathing is hard, noisy, or fast",
                "If you feel weak, dizzy, or unable to eat properly",
            ],
            "doctor_when_hi": [
                "अगर खांसी और बुखार जारी रहें या बढ़ जाएं।",
                "अगर सांस लेना कठिन, तेज या आवाज के साथ हो।",
                "अगर बहुत कमजोरी, चक्कर या ठीक से खाना न खा पाएं।",
            ],
        },
    }

    details = report_by_label.get(label, report_by_label["PNEUMONIA"]).copy()
    details.update(common)
    details["filename"] = filename
    details["confidence_percent"] = confidence_percent
    details["normalized_label"] = label
    return details


def get_localized_report_content(report, lang="en"):
    lang = "hi" if lang == "hi" else "en"
    suffix = "_hi" if lang == "hi" else ""
    labels = {
        "title": "पूरी हेल्थ गाइडेंस रिपोर्ट" if lang == "hi" else "Full Health Guidance Report",
        "subtitle": "एआई आधारित स्क्रीनिंग रिपोर्ट और अगले कदम" if lang == "hi" else "AI-assisted screening report and next steps",
        "prediction_label": "अनुमान" if lang == "hi" else "Prediction",
        "risk_label": "जोखिम स्तर" if lang == "hi" else "Risk Level",
        "confidence_label": "विश्वास" if lang == "hi" else "Confidence",
        "image_label": "इमेज" if lang == "hi" else "Image",
        "summary_title": "सारांश" if lang == "hi" else "Summary",
        "next_title": "अब आगे क्या करें" if lang == "hi" else "What To Do Next",
        "food_title": "क्या खाएं / क्या बेहतर है" if lang == "hi" else "What To Eat / Prefer",
        "avoid_title": "क्या अवॉइड करें" if lang == "hi" else "What To Avoid",
        "care_title": "किन बातों का ध्यान रखें" if lang == "hi" else "Things To Take Care Of",
        "doctor_title": "डॉक्टर के पास कब जाएं" if lang == "hi" else "When To See A Doctor",
        "warning_title": "इमरजेंसी चेतावनी संकेत" if lang == "hi" else "Emergency Warning Signs",
        "disclaimer_title": "महत्वपूर्ण सूचना" if lang == "hi" else "Important Note",
        "generated_note": "यह रिपोर्ट मौजूदा भाषा के अनुसार डाउनलोड की गई है।" if lang == "hi" else "This report was downloaded in the currently selected language.",
        "footer": "यह अंतिम मेडिकल डायग्नोसिस नहीं है। डॉक्टर से पुष्टि जरूरी है।" if lang == "hi" else "This is not a final medical diagnosis. Please confirm with a doctor.",
    }

    return {
        "lang": lang,
        "labels": labels,
        "display_label": report[f"display_label{suffix}"] if suffix else report["display_label"],
        "risk_level": report[f"risk_level{suffix}"] if suffix else report["risk_level"],
        "summary": report[f"summary{suffix}"] if suffix else report["summary"],
        "next_steps": report[f"next_steps{suffix}"] if suffix else report["next_steps"],
        "eat_more": report[f"eat_more{suffix}"] if suffix else report["eat_more"],
        "avoid": report[f"avoid{suffix}"] if suffix else report["avoid"],
        "general_care": report[f"general_care{suffix}"] if suffix else report["general_care"],
        "doctor_when": report[f"doctor_when{suffix}"] if suffix else report["doctor_when"],
        "warning_signs": report[f"warning_signs{suffix}"] if suffix else report["warning_signs"],
        "disclaimer": report[f"disclaimer{suffix}"] if suffix else report["disclaimer"],
        "filename": report.get("filename") or "N/A",
        "confidence_percent": report["confidence_percent"],
    }


def _load_pdf_font(size, bold=False):
    font_path = PDF_FONT_BOLD_PATH if bold and PDF_FONT_BOLD_PATH.exists() else PDF_FONT_PATH
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _draw_wrapped_text(draw, text, font, fill, x, y, max_width, line_gap=10, bullet=False):
    prefix = "• " if bullet else ""
    words = (prefix + (text or "")).split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = [prefix.strip()]

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def generate_report_pdf(report, lang="en"):
    content = get_localized_report_content(report, lang)
    page_width, page_height = 1240, 1754
    margin_x, margin_y = 72, 74
    card_padding = 34
    section_gap = 18
    body_color = "#17304a"
    muted_color = "#5d748d"
    accent_color = "#0f6bdc"
    accent_soft = "#eaf3ff"
    surface_color = "#ffffff"
    border_color = "#d9e6f4"
    bg_color = "#f3f8fe"

    title_font = _load_pdf_font(42, bold=True)
    heading_font = _load_pdf_font(28, bold=True)
    subheading_font = _load_pdf_font(22, bold=True)
    body_font = _load_pdf_font(21)
    small_font = _load_pdf_font(18)

    pages = []
    page = Image.new("RGB", (page_width, page_height), bg_color)
    draw = ImageDraw.Draw(page)
    current_y = margin_y

    def start_new_page():
        nonlocal page, draw, current_y
        pages.append(page)
        page = Image.new("RGB", (page_width, page_height), bg_color)
        draw = ImageDraw.Draw(page)
        current_y = margin_y
        draw_header()

    def ensure_space(height_needed):
        nonlocal current_y
        if current_y + height_needed > page_height - 110:
            start_new_page()

    def rounded_card(x1, y1, x2, y2, fill=surface_color, outline=border_color, radius=26):
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=fill, outline=outline, width=2)

    def draw_header():
        nonlocal current_y
        draw.rounded_rectangle((margin_x, current_y, page_width - margin_x, current_y + 150), radius=34, fill="#0d5fd1")
        draw.text((margin_x + 36, current_y + 26), "MediScan AI", font=heading_font, fill="#ffffff")
        draw.text((margin_x + 36, current_y + 74), content["labels"]["title"], font=title_font, fill="#ffffff")
        draw.text((margin_x + 36, current_y + 122), content["labels"]["subtitle"], font=small_font, fill="#dcecff")
        current_y += 182

    def draw_metrics():
        nonlocal current_y
        card_height = 250
        ensure_space(card_height + 10)
        rounded_card(margin_x, current_y, page_width - margin_x, current_y + card_height)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], report["filename"]) if report.get("filename") else ""
        preview_box = (margin_x + 28, current_y + 28, margin_x + 310, current_y + 222)
        draw.rounded_rectangle(preview_box, radius=24, fill=accent_soft)
        if image_path and os.path.exists(image_path):
            try:
                preview = Image.open(image_path).convert("RGB")
                preview.thumbnail((preview_box[2] - preview_box[0], preview_box[3] - preview_box[1]))
                px = preview_box[0] + ((preview_box[2] - preview_box[0]) - preview.width) // 2
                py = preview_box[1] + ((preview_box[3] - preview_box[1]) - preview.height) // 2
                page.paste(preview, (px, py))
            except Exception:
                draw.text((preview_box[0] + 32, preview_box[1] + 74), content["labels"]["image_label"], font=subheading_font, fill=accent_color)
        else:
            draw.text((preview_box[0] + 32, preview_box[1] + 74), content["labels"]["image_label"], font=subheading_font, fill=accent_color)

        info_x = margin_x + 356
        draw.text((info_x, current_y + 34), f"{content['labels']['prediction_label']}: {content['display_label']}", font=subheading_font, fill=body_color)
        draw.text((info_x, current_y + 82), f"{content['labels']['risk_label']}: {content['risk_level']}", font=body_font, fill=muted_color)
        draw.text((info_x, current_y + 120), f"{content['labels']['confidence_label']}: {content['confidence_percent']}%", font=body_font, fill=muted_color)
        draw.text((info_x, current_y + 158), f"{content['labels']['image_label']}: {content['filename']}", font=small_font, fill=muted_color)

        badge_text = content["risk_level"]
        badge_width = int(draw.textlength(badge_text, font=small_font)) + 54
        draw.rounded_rectangle((page_width - margin_x - badge_width, current_y + 34, page_width - margin_x - 24, current_y + 84), radius=22, fill=accent_soft)
        draw.text((page_width - margin_x - badge_width + 22, current_y + 47), badge_text, font=small_font, fill=accent_color)
        current_y += card_height + 18

    def draw_section(title, items, single_paragraph=False):
        nonlocal current_y
        estimated_height = 95 + (54 * max(len(items), 1))
        if single_paragraph:
            estimated_height = 180
        ensure_space(estimated_height)
        section_top = current_y
        section_bottom = current_y + estimated_height
        rounded_card(margin_x, section_top, page_width - margin_x, section_bottom)
        draw.text((margin_x + card_padding, section_top + 24), title, font=heading_font, fill=body_color)
        y = section_top + 76
        max_width = page_width - (margin_x * 2) - (card_padding * 2)
        if single_paragraph:
            y = _draw_wrapped_text(draw, items[0], body_font, muted_color, margin_x + card_padding, y, max_width, line_gap=12)
        else:
            for item in items:
                y = _draw_wrapped_text(draw, item, body_font, muted_color, margin_x + card_padding, y, max_width, line_gap=10, bullet=True)
                y += 4
        current_y = max(y + 22, section_bottom + section_gap)

    draw_header()
    draw_metrics()
    draw_section(content["labels"]["summary_title"], [content["summary"]], single_paragraph=True)
    draw_section(content["labels"]["next_title"], content["next_steps"])
    draw_section(content["labels"]["food_title"], content["eat_more"])
    draw_section(content["labels"]["avoid_title"], content["avoid"])
    draw_section(content["labels"]["care_title"], content["general_care"])
    draw_section(content["labels"]["doctor_title"], content["doctor_when"])
    draw_section(content["labels"]["warning_title"], content["warning_signs"])
    draw_section(content["labels"]["disclaimer_title"], [content["disclaimer"], content["labels"]["generated_note"]])

    footer_y = page_height - 72
    draw.text((margin_x, footer_y), content["labels"]["footer"], font=small_font, fill=muted_color)
    pages.append(page)

    pdf_bytes = BytesIO()
    rgb_pages = [img.convert("RGB") for img in pages]
    rgb_pages[0].save(pdf_bytes, format="PDF", save_all=True, append_images=rgb_pages[1:])
    return pdf_bytes.getvalue()

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image_batch = build_tta_batch(image)
    predictions = model.predict(image_batch, verbose=0)
    labels = le.inverse_transform(np.arange(len(predictions[0])))
    summary = summarize_prediction(predictions, labels)
    return summary

def is_valid_xray(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False

    focus_region, roi_meta = extract_focus_roi(image)
    image = focus_region if focus_region is not None and focus_region.size else image

    height, width = image.shape[:2]
    if height < 120 or width < 120:
        return False

    aspect_ratio = width / float(height)
    if aspect_ratio < 0.6 or aspect_ratio > 1.45:
        return False

    image_float = image.astype("float32")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channel_diff = float(
        np.mean(np.abs(image_float[:, :, 0] - image_float[:, :, 1])) +
        np.mean(np.abs(image_float[:, :, 1] - image_float[:, :, 2]))
    )
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    mean_intensity = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    dark_ratio = float(np.mean(gray < 55))
    bright_ratio = float(np.mean(gray > 200))

    center_crop = gray[int(height * 0.12):int(height * 0.9), int(width * 0.12):int(width * 0.88)]
    if center_crop.size == 0 or center_crop.shape[1] < 40:
        return False

    mid = center_crop.shape[1] // 2
    left_half = center_crop[:, :mid]
    right_half = cv2.flip(center_crop[:, center_crop.shape[1] - mid:], 1)

    if left_half.size == 0 or right_half.size == 0:
        return False

    left_norm = cv2.normalize(left_half.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)
    right_norm = cv2.normalize(right_half.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)
    symmetry_score = float(np.mean(1.0 - np.abs(left_norm - right_norm)))

    grayscale_like = channel_diff < 22 and mean_saturation < 32
    medical_exposure = 85 <= mean_intensity <= 210
    medical_contrast = 28 <= contrast <= 105
    balanced_tones = dark_ratio > 0.04 and bright_ratio > 0.02 and (dark_ratio + bright_ratio) < 0.72
    structured_image = 0.012 <= edge_density <= 0.22
    chest_like_symmetry = symmetry_score >= 0.60
    histogram = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    histogram /= max(float(histogram.sum()), 1.0)
    midtone_balance = float(np.sum(histogram[3:13]))
    extreme_tones = float(histogram[0] + histogram[1] + histogram[-1] + histogram[-2])
    medically_distributed = midtone_balance > 0.52 and extreme_tones < 0.42

    return all([
        grayscale_like,
        medical_exposure,
        medical_contrast,
        balanced_tones,
        structured_image,
        chest_like_symmetry,
        medically_distributed,
    ]) and (roi_meta["detected"] or edge_density >= 0.025)


@app.errorhandler(413)
def request_entity_too_large(_error):
    return render_invalid_result(message='File is too large. Please upload a chest X-ray image under 10MB.'), 413

@app.route('/')
def home():
    return render_page('index.html', 'home')

@app.route('/about')
def about():
    return render_page('about.html', 'about')

@app.route('/news')
def news():
    return render_page('news.html', 'news')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_invalid_result(message='No image received. Please upload a chest X-ray image first.')

    file = request.files['file']
    if file.filename == '':
        return render_invalid_result(message='No image selected. Please choose a chest X-ray image first.')

    if not allowed_file_extension(file.filename):
        return render_invalid_result(message='Unsupported file type. Please use JPG, JPEG, or PNG chest X-ray images only.')

    if file:
        original_name = secure_filename(file.filename)
        name, ext = os.path.splitext(original_name)
        filename = f"{name or 'scan'}_{uuid4().hex[:8]}{ext or '.jpg'}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            if not is_valid_xray(file_path):
                return render_invalid_result(filename=filename)
            prediction = process_image(file_path)
            return render_template('result.html',
                                   image_path=file_path,
                                   filename=filename,
                                   predicted_label=prediction["predicted_label"],
                                   confidence_score=prediction["confidence_score"],
                                   confidence_margin=prediction["margin"],
                                   is_uncertain=prediction["is_uncertain"])
        except Exception:
            remove_file_safely(file_path)
            return render_invalid_result(message='Unable to process the image. Please try again with a clear chest X-ray.')

    return render_invalid_result(message='Unable to process the image. Please try again with a chest X-ray.')

@app.route('/camera')
def camera():
    return render_page('camera.html', 'camera')

@app.route('/report')
def report():
    filename = request.args.get('filename', '').strip()
    predicted_label = request.args.get('label', '').strip()
    lang = 'hi' if request.args.get('lang', '').strip().lower() == 'hi' else 'en'
    try:
        confidence_score = float(request.args.get('confidence', '0'))
    except ValueError:
        confidence_score = 0.0

    report_data = build_report_context(filename, predicted_label, confidence_score)
    return render_template('report.html', report=report_data, active_page='', current_lang=lang)

@app.route('/report/pdf')
def report_pdf():
    filename = request.args.get('filename', '').strip()
    predicted_label = request.args.get('label', '').strip()
    lang = 'hi' if request.args.get('lang', '').strip().lower() == 'hi' else 'en'
    try:
        confidence_score = float(request.args.get('confidence', '0'))
    except ValueError:
        confidence_score = 0.0

    report_data = build_report_context(filename, predicted_label, confidence_score)
    pdf_bytes = generate_report_pdf(report_data, lang=lang)
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    safe_name = (filename or 'report').replace(' ', '_')
    response.headers['Content-Disposition'] = f'attachment; filename={safe_name}_full_report_{lang}.pdf'
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return {"reply": "Please enter a question."}

    bot_reply = medical_chatbot(user_message)

    return {"reply": bot_reply}

if __name__ == '__main__':
    app.run(debug=True)
