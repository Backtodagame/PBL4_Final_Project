import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import math
from collections import Counter
from torchvision import models, transforms

# ==========================================
# 1. KHAI BÁO MÔ HÌNH TINH GỌN (PURE RESNET-18)
# ==========================================
class PureResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(PureResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False) # Quá trình test không cần tải lại Pre-trained
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ==========================================
# 2. CẤU HÌNH & THIẾT BỊ
# ==========================================
# LƯU Ý: Đổi đường dẫn tới file bạn vừa tải từ Kaggle về
#'model_resnet.pth
#Model1 = 'model_resnet.pth'
#model2= 'pure_resnet_best.pth'

CHECKPOINT_PATH = 'nb_resnet_best.pth'
class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_webcam_model():
    model = PureResNet18(num_classes=7).to(device)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # Tùy thuộc vào cách lưu, file pth có thể chứa model_state_dict hoặc chứa luôn weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Đã load thành công 'bộ não' trên {device}!")
        return model
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        exit()

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# 3. TÍNH TOÁN BÙ TRỪ THIÊN KIẾN (LOG-SCALED BIAS)
# ==========================================
CLASS_COUNTS = {
    'Surprise': 1290, 'Fear': 281, 'Disgust': 717,
    'Happiness': 4772, 'Sadness': 1982, 'Anger': 705, 'Neutral': 2524
}

def calculate_mean_log_bias(counts_dict, alpha=0.15):
    total_images = sum(counts_dict.values())
    average_count = total_images / len(counts_dict)
    return {k: alpha * math.log(average_count / v) for k, v in counts_dict.items()}

# Khởi tạo Tensor bù trừ và đưa lên GPU
calibration_bias_dict = calculate_mean_log_bias(CLASS_COUNTS, alpha=0.15)
bias_list = [calibration_bias_dict[emo] for emo in class_names]
bias_tensor = torch.tensor(bias_list, device=device)

# ==========================================
# 4. VÒNG LẶP XỬ LÝ WEBCAM
# ==========================================
def run_webcam():
    model = load_webcam_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("--- Đang chạy Webcam... Nhấn 'q' để thoát ---")

    session_history = []
    current_second_preds = []
    start_time = time.time()
    display_label = "Detecting..."

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            input_tensor = data_transforms(face_rgb).unsqueeze(0).to(device)

            # 3. Dự đoán thuần túy (Không cần bù trừ nữa)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                
            # Không cần biến calibrated_probs hay bias_tensor nữa
            # Chọn thẳng nhãn có xác suất cao nhất từ mô hình
            pred_idx = torch.argmax(probs).item()
            label = class_names[pred_idx]
            current_second_preds.append(label)

            # Tính cảm xúc theo từng giây
            time_elapsed = time.time() - start_time
            if time_elapsed >= 1.0: 
                if current_second_preds:
                    most_common = Counter(current_second_preds).most_common(1)[0][0]
                    session_history.append(most_common)
                    display_label = most_common
                    print(f"⏱ Giây {len(session_history)}: {most_common}")
                current_second_preds = []
                start_time = time.time()

            # Hiển thị lên khung hình
            color = (0, 255, 0) 
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('PBL4: Face Emotion Recognition - Tinh Gon', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # In Báo cáo tổng kết (giữ nguyên logic)
    print("\n" + "="*40)
    print("📊 BÁO CÁO TỔNG KẾT PHIÊN WEBCAM")
    print("="*40)
    if session_history:
        final_counts = Counter(session_history)
        dom_emo, dom_count = final_counts.most_common(1)[0]
        print(f"🏆 CHỦ ĐẠO: {dom_emo.upper()} ({dom_count/len(session_history)*100:.1f}%)")
        for emo, c in final_counts.most_common():
            print(f" - {emo:<10}: {c} giây")
    print("="*40)

if __name__ == "__main__":
    run_webcam()