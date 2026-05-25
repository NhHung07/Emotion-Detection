# data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_data_loaders(train_dir, test_dir, batch_size=64, num_workers=2):
    """
    Hàm khởi tạo DataLoader cho bộ dữ liệu FER-2013.
    """
    # 1. Định nghĩa Transforms (Augmentation cho Train, giữ nguyên cho Test)
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Load Dataset từ thư mục
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    # 3. Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # Tối ưu hóa chuyển dữ liệu lên CUDA
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.classes