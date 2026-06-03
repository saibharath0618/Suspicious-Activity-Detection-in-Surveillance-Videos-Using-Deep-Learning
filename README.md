# 🎥 Suspicious Activity Detection in Surveillance Videos Using Deep Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-green?style=flat)

## 🔍 About
A deep learning-based system that **monitors campus CCTV footage** and automatically identifies suspicious or illegal activities. The system sends real-time alert messages to the head security person whenever unusual behavior is detected. The model was trained and tuned on real-world video data to improve detection accuracy and reliability.

## 💡 Key Highlights
- 🏫 Designed for **campus CCTV surveillance** monitoring
- 🤖 Uses **MobileNet + LSTM** for frame-level + temporal activity recognition
- 🚨 Automatic **real-time alert** sent to security personnel on detection
- 🎯 Model fine-tuned on real video data for high accuracy & low false positives
- 🌐 Web interface built with **Flask** for live monitoring
- 🗄️ User management via **SQLite** database

## 🧠 Model Architecture

| Stage | Component | Role |
|-------|-----------|------|
| 1 | **MobileNet** | Extracts features from individual video frames |
| 2 | **LSTM** | Models temporal patterns across frame sequences |
| Output | **Classification** | Flags suspicious vs normal activity |

## 🛠️ Tools Used
- **Python** — core programming
- **TensorFlow / Keras** — model training & inference
- **MobileNet + LSTM** — deep learning architecture
- **Flask** — web application interface
- **SQLite** — user database management
- **OpenCV** — video frame processing

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main Flask web application |
| `sus train.py` | Model training script |
| `sus test.py` | Suspicious activity testing script |
| `final test.py` | Final model evaluation script |
| `test.py` | General testing utilities |
| `mobilenet_lstm_generator.h5` | Trained Keras model weights |
| `best.pt` | Best model checkpoint |
| `classes.txt` | List of activity/behavior classes |
| `users.db` | SQLite user database |

## 🚀 How to Run
```bash
git clone https://github.com/saibharath0618/Suspicious-Activity-Detection-in-Surveillance-Videos-Using-Deep-Learning.git
cd Suspicious-Activity-Detection-in-Surveillance-Videos-Using-Deep-Learning
pip install -r requirements.txt
python app.py
```

To retrain the model:
```bash
python "sus train.py"
```

To evaluate performance:
```bash
python "final test.py"
```

## 🔔 Alert System Flow
```
CCTV Feed → Frame Extraction → MobileNet Features → LSTM Sequence → Detection → 🚨 Alert Sent
```

## 📅 Project Info
- **Domain:** Computer Vision / Surveillance
- **Architecture:** MobileNet + LSTM
- **Use Case:** Campus Security Monitoring
- **Training Data:** Real-world labeled surveillance video

## 🙋 About Me
**Patakanoor Sai Bharath** — Aspiring Data & AI Engineer  
Passionate about building intelligent systems that solve real-world problems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/patakanoor-sai-bharath-02643125b)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/saibharath0618)
