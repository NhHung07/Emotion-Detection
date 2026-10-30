# import thư viện cần thiết
import torch
from torchvision import datasets, transforms # Thư viện để xử lý ảnh và tạo dataset
from torch.utils.data import DataLoader # Thư viện để tạo DataLoader giúp load dữ liệu theo batch

def build_data_loaders(train_dir, test_dir, batch_size=64, num_workers=2):
    """
    Hàm khởi tạo DataLoader cho bộ dữ liệu FER-2013.
    """
    # 1. Định nghĩa Transforms (Augmentation cho Train, giữ nguyên cho Test)
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Chuyển thành ảnh xám với 1 kênh màu
        transforms.RandomHorizontalFlip(p=0.5), # Lât ngang với xác suất 50% giúp nhận diện ảnh bị lật
        transforms.RandomRotation(10), # Xoay ngẫu nhiên trong khoảng -10 đến 10 độ giúp nhận diện ảnh nghiêng đầu
        transforms.ToTensor(), # chuyển dữ liệu ảnh từ PIL Image sang Tensor để đưa vào mô hình
        transforms.Normalize(mean=[0.5], std=[0.5]) # Chuẩn hóa ảnh về khoảng [-1, 1] giúp mô hình học tốt hơn
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Chuyển thành ảnh xám với 1 kênh màu
        transforms.ToTensor(), # chuyển dữ liệu ảnh từ PIL Image sang Tensor để đưa vào mô hình
        transforms.Normalize(mean=[0.5], std=[0.5]) # Chuẩn hóa ảnh về khoảng [-1, 1]
    ])

    # 2. Load Dataset từ thư mục với ImageFolder, tự động gán nhãn dựa trên tên thư mục con
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    # 3. Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset, # Dataset train chứa ảnh và nhãn đã được xử lý
        batch_size=batch_size, # Số lượng ảnh trong mỗi batch giúp mô hình học hiệu quả hơn
        shuffle=True, # Trộn dữ liệu mỗi epoch giúp mô hình học tốt hơn
        num_workers=num_workers, # Sử dụng đa luồng để tăng tốc độ load dữ liệu
        pin_memory=True # Tối ưu hóa chuyển dữ liệu lên CUDA
    )

    test_loader = DataLoader(
        test_dataset, # Dataset test chứa ảnh và nhãn đã được xử lý
        batch_size=batch_size, # Số lượng ảnh trong mỗi batch giúp mô hình đánh giá hiệu quả hơn
        shuffle=False, # Không trộn dữ liệu để đảm bảo tính nhất quán trong quá trình đánh giá
        num_workers=num_workers, # Sử dụng đa luồng để tăng tốc độ load dữ liệu
        pin_memory=True # Tối ưu hóa chuyển dữ liệu lên CUDA
    )

    return train_loader, test_loader, train_dataset.classes