import pathlib
import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk

# ===============================
# WINDOWS PATH FIX
# ===============================
pathlib.PosixPath = pathlib.WindowsPath

# ===============================
# LOAD YOLOv5 MODEL
# ===============================
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='best.pt',
    force_reload=False
)

model.conf = 0.05
model.iou = 0.5
model.max_det = 10   # ✅ MULTIPLE ID CARDS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===============================
# TKINTER UI
# ===============================
root = tk.Tk()
root.title("Real-Time YOLOv5 Multi-ID Detection")

video_label = tk.Label(root)
video_label.pack()

running = True

# ===============================
# TRACKING MEMORY
# ===============================
tracked_boxes = []
missed_frames = 0
MAX_MISSED_FRAMES = 20

# ===============================
# DETECTION LOOP
# ===============================
def detect():
    global tracked_boxes, missed_frames

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, detect)
        return

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, size=640)

    current_boxes = []

    if len(results.xyxy[0]) > 0:
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            label = results.names[int(cls)]

            if label == "id_card":
                x1, y1, x2, y2 = map(int, box)
                current_boxes.append((x1, y1, x2, y2, conf))

    if current_boxes:
        tracked_boxes = current_boxes
        missed_frames = 0
    else:
        missed_frames += 1
        if missed_frames > MAX_MISSED_FRAMES:
            tracked_boxes = []

    # ===============================
    # DRAW ALL BOXES
    # ===============================
    for (x1, y1, x2, y2, conf) in tracked_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"id_card {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.config(image=imgtk)
    video_label.image = imgtk

    root.after(10, detect)

# ===============================
# STOP
# ===============================
def stop():
    global running
    running = False
    cap.release()
    root.destroy()

tk.Button(
    root,
    text="STOP",
    command=stop,
    bg="red",
    fg="white",
    font=("Arial", 12, "bold")
).pack(pady=10)

detect()
root.mainloop()
