# Facial Expression Recognition (FER)

Dự án nhận diện 7 cảm xúc khuôn mặt theo thời gian thực qua Webcam sử dụng PyTorch và OpenCV:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

---

## 1. Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

## 2. Cấu trúc thư mục

```text
├── data/                         # Dữ liệu FER-2013 (train/ và test/)
├── emojis/                       # Ảnh emoji dán đè lên webcam
├── data_loader.py                # Xử lý và nạp dữ liệu
├── model.py                      # Kiến trúc EmotionCNN và ResNet-18
├── train_engine.py               # Hàm huấn luyện mô hình
├── experiment.ipynb              # Notebook huấn luyện và vẽ đồ thị
├── webcam.py                     # Chạy nhận diện trực tiếp qua Webcam
├── emotion_cnn_weights.pth       # Trọng số EmotionCNN
├── resnet18_emotion_weights.pth  # Trọng số ResNet-18
└── requirements.txt              # Danh sách thư viện
```

---

## 3. Chạy Demo Webcam

Chạy nhận diện trực tiếp qua camera:

```bash
python webcam.py
```

- Bấm phím **`q`** trên cửa sổ hiển thị để thoát.
- Mặc định mô hình đang sử dụng là `ResNetEmotion`. Để chuyển sang `EmotionCNN`, bỏ comment phần cấu hình model trong file `webcam.py`.

---

## 4. Huấn luyện mô hình

Mở file `experiment.ipynb` bằng Jupyter Notebook và chạy lần lượt các cell để:
1. Nạp và tăng cường dữ liệu từ `data/train` và `data/test`.
2. Khởi tạo mô hình (`EmotionCNN` hoặc `ResNetEmotion`).
3. Huấn luyện với optimizer Adam, scheduler và lưu file trọng số `.pth`.
