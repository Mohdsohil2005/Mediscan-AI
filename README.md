# 🩺 MediScan AI

### AI-Powered X-Ray Analysis System using Deep Learning

**MediScan AI** is a smart healthcare web application that leverages **Convolutional Neural Networks (CNN)** to analyze chest X-ray images and detect potential medical conditions such as COVID-19.

Designed with a clean UI and real-time processing, this project demonstrates the integration of **AI + Web Development + Computer Vision**.

---

## 🌟 Key Highlights

* 🤖 Deep Learning-based X-ray classification (CNN Model)
* 📷 Real-time image capture using camera
* 🎯 ROI-based scanning for focused detection
* ⚡ Fast and optimized prediction pipeline
* 🌐 Fully responsive web interface
* 🌙 Dark mode supported UI
* ☁️ Deployed on cloud (Render)

---

## 🧠 Working Pipeline

```text
User Input (Camera / Upload)
        ↓
Image Preprocessing (Resize, Normalize)
        ↓
CNN Model Prediction
        ↓
Result Output (Prediction Display)
```

---

## 🛠️ Tech Stack

| Layer          | Technology Used          |
| -------------- | ------------------------ |
| Frontend       | HTML, CSS, JavaScript    |
| Backend        | Flask (Python)           |
| AI/ML Model    | TensorFlow / Keras (CNN) |
| Image Handling | OpenCV                   |
| Deployment     | Render                   |

---

## 📂 Project Structure

```
mediscan-ai/
│
├── app.py                # Main Flask application
├── train_model.py       # Model training script
├── ml_utils.py          # ML helper functions
├── models/              # Trained model (ignored in Git)
├── static/              # CSS, JS, Images
├── templates/           # HTML templates
├── uploads/             # User uploaded images
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/mediscan-ai.git
cd mediscan-ai
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Application

```
python app.py
```

---

## 🌐 Live Demo

🔗 https://mediscan-ai-rla4.onrender.com/

---

## 📸 Screenshots

> Add UI screenshots here (Home page, Upload page, Result page)

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**.
It should **not be used as a substitute for professional medical diagnosis**.

---

## 🚀 Future Enhancements

* 📊 Detailed medical report generation
* 📄 PDF export functionality
* 🔍 Multi-disease detection
* ☁️ Model hosting via API (HuggingFace / AWS)
* 🔐 User authentication system

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

---

## 👨‍💻 Author

**Mohd Sohil**
📩 Email:[sohil.jaipur25@gmail.com]
🌐 Portfolio: https://www.linkedin.com/in/mohd-sohil-4a4b2a277

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
