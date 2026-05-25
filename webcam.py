# webcam.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import EmotionCNN
import os

# ==========================================
# 1. CẤU HÌNH THIẾT BỊ VÀ TẢI MÔ HÌNH
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load('emotion_cnn_weights.pth', weights_only=True))
model.eval()

# ==========================================
# 2. CHUẨN BỊ TRANSFORM & LOAD ẢNH EMOJI
# ==========================================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emojis = {}
emoji_size = 100 # Cố định kích thước Emoji to (100x100)

print("Đang nạp dữ liệu Emoji...")
for name in class_names:
    img_path = f"emojis/{name}.png"
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGBA")
        img = img.resize((emoji_size, emoji_size))
        emojis[name] = img
    else:
        emojis[name] = None

# ==========================================
# 3. MỞ WEBCAM VÀ DÒ TÌM KHUÔN MẶT
# ==========================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Đang mở Webcam... Bấm phím 'q' trên cửa sổ video để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        pil_roi = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        tensor_img = transform(pil_roi).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            _, predicted = torch.max(outputs, 1)
            emotion_label = class_names[predicted.item()]
            
        # ====================================================
        # BƯỚC 3.1: DÁN EMOJI (Làm trước để không đè lên chữ)
        # ====================================================
        icon = emojis.get(emotion_label)
        if icon is not None:
            # Chuyển frame gốc sang PIL
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            paste_x = x + w + 15
            paste_y = y
            if paste_x + emoji_size < frame_pil.width:
                frame_pil.paste(icon, (paste_x, paste_y), mask=icon)
            
            # Ép ngược lại frame của OpenCV sau khi đã dán xong Emoji
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # ====================================================
        # BƯỚC 3.2: VẼ KHUNG & CHỮ LÊN KHUÔN MẶT THẬT
        # ====================================================
        # Vẽ khung xanh lá
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Căn chữ ngay sát trên đỉnh khung nhận diện (y - 15)
        text_y = y - 15 if y - 15 > 20 else y + 20 
        cv2.putText(frame, emotion_label.upper(), (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow('Facial Expression Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()