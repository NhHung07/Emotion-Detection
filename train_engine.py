# train_engine.py
import torch
from tqdm import tqdm # Tạo thanh tiến trình (progress bar) cho console

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, scheduler=None):
    """
    Hàm huấn luyện và đánh giá mô hình qua từng Epoch.
    Trả về dictionary chứa lịch sử Loss và Accuracy để vẽ biểu đồ.
    """
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # ==========================================
        # 1. GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
        # ==========================================
        model.train() # Bật chế độ train (Kích hoạt Dropout và BatchNorm)
        running_loss = 0.0
        
        # Tích hợp thanh tiến trình cho tập train
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for images, labels in loop:
            # Đẩy dữ liệu lên GPU
            images, labels = images.to(device), labels.to(device)
            
            # 5 bước tối ưu hóa của PyTorch
            optimizer.zero_grad()              # 1. Xóa gradient cũ
            outputs = model(images)            # 2. Forward pass
            loss = criterion(outputs, labels)  # 3. Tính độ lỗi (Loss)
            loss.backward()                    # 4. Backward pass (Lan truyền ngược)
            optimizer.step()                   # 5. Cập nhật trọng số (Weights)
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # ==========================================
        # 2. GIAI ĐOẠN ĐÁNH GIÁ (VALIDATION/TESTING PHASE)
        # ==========================================
        model.eval() # Chuyển sang chế độ đánh giá (Tắt Dropout)
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # Tắt tính toán gradient để tăng tốc và giảm RAM
            for images, labels in test_loader:
                # Đẩy dữ liệu lên GPU
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Tìm class có xác suất cao nhất
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        
        # ==========================================
        # 3. CẬP NHẬT SCHEDULER (Phần mới thêm)
        # ==========================================
        # Lấy Learning Rate hiện tại để in ra màn hình
        current_lr = optimizer.param_groups[0]['lr']
        
        # Nếu có truyền scheduler vào, cho nó cập nhật dựa trên Test Loss
        if scheduler is not None:
            scheduler.step(avg_test_loss)

        # In tổng kết mỗi Epoch
        print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
    return history