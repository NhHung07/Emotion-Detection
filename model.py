# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # ==========================================
        # BLOCK 1: Trích xuất đặc trưng cơ bản (Cạnh, góc)
        # Input: [Batch, 1, 48, 48] -> Output: [Batch, 32, 24, 24]
        # ==========================================
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Giúp hội tụ nhanh hơn, giảm lệ thuộc vào init weight
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==========================================
        # BLOCK 2: Trích xuất đặc trưng bậc trung (Kết cấu)
        # Input: [Batch, 32, 24, 24] -> Output: [Batch, 64, 12, 12]
        # ==========================================
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==========================================
        # BLOCK 3: Trích xuất đặc trưng bậc cao (Các bộ phận khuôn mặt)
        # Input: [Batch, 64, 12, 12] -> Output: [Batch, 128, 6, 6]
        # ==========================================
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==========================================
        # BLOCK 4: Phân loại (Fully Connected Layers)
        # Kích thước tensor sau Block 3: 128 channels * 6 height * 6 width = 4608 features
        # ==========================================
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout = nn.Dropout(0.5) # Tắt ngẫu nhiên 50% neuron để chống Overfitting
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Truyền tiến qua các Block Tích chập
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Dàn phẳng tensor từ 4D [Batch, C, H, W] về 2D [Batch, Features]
        x = x.view(x.size(0), -1)
        
        # Truyền qua lớp Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output cuối cùng không dùng Softmax ở đây (vì hàm Loss sẽ tự gộp)
        
        return x
    
class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()
        
        # 1. Tải kiến trúc ResNet18 đã được Pre-trained
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Xử lý "nút thắt" kênh màu: 
        # ResNet18 mặc định nhận ảnh màu RGB (3 kênh). Ảnh FER-2013 của ta là ảnh xám (1 kênh).
        # Ta cần thay thế lớp Tích chập đầu tiên (conv1) để nó nhận 1 kênh.
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, # Đổi thành 1
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        # Mẹo tối ưu: Lấy trung bình cộng trọng số của 3 kênh màu cũ đắp sang kênh xám mới
        with torch.no_grad():
            self.resnet.conv1.weight[:] = torch.mean(original_conv.weight, dim=1, keepdim=True)
        
        # 3. Thay đổi lớp Phân loại cuối cùng (Fully Connected)
        # ResNet18 mặc định phân loại 1000 đồ vật, ta ép nó về 7 cảm xúc
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), # Kẹp thêm Dropout để chống Overfitting
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)