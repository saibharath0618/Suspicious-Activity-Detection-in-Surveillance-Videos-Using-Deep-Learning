import pathlib
import cv2
import torch
import numpy as np
import tensorflow as tf
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =====================================================
# WINDOWS PATH FIX
# =====================================================
pathlib.PosixPath = pathlib.WindowsPath

# =====================================================
# EMAIL CONFIG
# =====================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"

def send_alert_email(conf):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "⚠️ Suspicious Activity Detected"

        body = f"""
Suspicious activity detected!

Confidence: {conf:.3f}

Please check the live surveillance feed immediately.
"""
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("📧 Alert email sent")

    except Exception as e:
        print("❌ Email error:", e)

# =====================================================
# SCREEN DETECTION (STATIC SCREEN)
# =====================================================
def screen_present(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size

    if brightness > 170 and edge_ratio < 0.04:
        return True
    return False

# =====================================================
# VIDEO ON SCREEN DETECTION (ANTI-SPOOF)
# =====================================================
prev_gray = None
motion_buffer = deque(maxlen=5)

def video_playing(frame):
    global prev_gray

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        return False

    diff = cv2.absdiff(prev_gray, gray)
    motion_score = np.mean(diff)

    prev_gray = gray
    motion_buffer.append(motion_score)

    avg_motion = np.mean(motion_buffer)

    # tuned threshold
    if avg_motion > 6.5:
        return True
    return False

# =====================================================
# YOLOv5 – ID CARD DETECTION
# =====================================================
id_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="best.pt",
    force_reload=False
)

id_model.conf = 0.05
id_model.iou = 0.5
id_model.max_det = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
id_model.to(device)
id_model.eval()

# =====================================================
# SUSPICIOUS ACTIVITY MODEL
# =====================================================
IMG_SIZE = 224
SEQ_LEN = 25
THRESHOLD = 0.60
MODEL_PATH = "mobilenet_lstm_generator.h5"

LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}

activity_model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Activity model loaded")

frame_buffer = deque(maxlen=SEQ_LEN)
confidence_buffer = deque(maxlen=5)
email_sent = False

# =====================================================
# WEBCAM
# =====================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("🎥 System started (Press Q to quit)")

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    screen_flag = screen_present(frame)
    video_flag = video_playing(frame)

    # -------------------------------------------------
    # ID CARD DETECTION (BLOCKED FOR SCREEN OR VIDEO)
    # -------------------------------------------------
    if not (screen_flag or video_flag):

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = id_model(img_rgb, size=640)

        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            label = results.names[int(cls)]
            if label == "id_card":
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    display,
                    f"id_card {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


    # -------------------------------------------------
    # ACTIVITY DETECTION
    # -------------------------------------------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    frame_buffer.append(img)

    status_text = "Collecting frames..."
    status_color = (0, 255, 255)

    if len(frame_buffer) == SEQ_LEN:
        seq = np.expand_dims(frame_buffer, axis=0)
        conf = activity_model.predict(seq, verbose=0)[0][0]

        confidence_buffer.append(conf)
        avg_conf = np.mean(confidence_buffer)

        if avg_conf >= THRESHOLD:
            status_text = f"SUSPICIOUS | Conf: {avg_conf:.3f}"
            status_color = (0, 0, 255)

            if not email_sent:
                send_alert_email(avg_conf)
                email_sent = True
        else:
            status_text = f"NORMAL | Conf: {avg_conf:.3f}"
            status_color = (0, 255, 0)
            email_sent = False

    cv2.putText(
        display,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.imshow("ID Card + Activity Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================================
# CLEANUP
# =====================================================
cap.release()
cv2.destroyAllWindows()
print("🛑 System stopped")
