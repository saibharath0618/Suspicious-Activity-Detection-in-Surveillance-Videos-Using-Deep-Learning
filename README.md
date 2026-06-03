🎥 Suspicious Activity Detection in Surveillance Videos Using Deep Learning
🔍 About
A deep learning-based system that monitors campus CCTV footage and automatically identifies suspicious or illegal activities. The system sends real-time alert messages to the head security person whenever unusual behavior is detected. The model was trained and tuned on real-world video data to improve detection accuracy and reliability.
💡 Key Highlights

🏫 Designed for campus CCTV surveillance monitoring
🤖 Uses MobileNet + LSTM for frame-level + temporal activity recognition
🚨 Automatic real-time alert sent to security personnel on detection
🎯 Model fine-tuned on real video data for high accuracy & low false positives
🌐 Web interface built with Flask for live monitoring
🗄️ User management via SQLite database

🧠 Model Architecture
StageComponentRole1MobileNetExtracts features from individual video frames2LSTMModels temporal patterns across frame sequencesOutputClassificationFlags suspicious vs normal activity
🛠️ Tools Used

Python — core programming
TensorFlow / Keras — model training & inference
MobileNet + LSTM — deep learning architecture
Flask — web application interface
SQLite — user database management
OpenCV — video frame processing

📁 Files
FileDescriptionapp.pyMain Flask web applicationsus train.pyModel training scriptsus test.pySuspicious activity testing scriptfinal test.pyFinal model evaluation scripttest.pyGeneral testing utilitiesmobilenet_lstm_generator.h5Trained Keras model weightsbest.ptBest model checkpointclasses.txtList of activity/behavior classesusers.dbSQLite user database
🚀 How to Run
bashgit clone https://github.com/saibharath0618/Suspicious-Activity-Detection-in-Surveillance-Videos-Using-Deep-Learning.git
cd Suspicious-Activity-Detection-in-Surveillance-Videos-Using-Deep-Learning
pip install -r requirements.txt
python app.py
To retrain the model:
bashpython "sus train.py"
To evaluate performance:
bashpython "final test.py"
🔔 Alert System Flow
CCTV Feed → Frame Extraction → MobileNet Features → LSTM Sequence → Detection → 🚨 Alert Sent
📅 Project Info

Domain: Computer Vision / Surveillance
Architecture: MobileNet + LSTM
Use Case: Campus Security Monitoring
Training Data: Real-world labeled surveillance video

🙋 About Me
Patakanoor Sai Bharath — Aspiring Data & AI Engineer
Passionate about building intelligent systems that solve real-world problems.
