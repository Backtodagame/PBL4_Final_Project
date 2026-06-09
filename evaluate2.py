"""
Mã nguồn thực nghiệm tự động v8.1 — Tích hợp xuất ảnh Báo cáo Đồ án (Dashboard)
Sử dụng dữ liệu chính diện RAVDESS kết hợp Inject câu chữ tiếng Việt.
"""

import os
import pandas as pd
import numpy as np
import cv2
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Nạp các hàm lõi từ file v8 của nhóm bạn
from mental_health_ui_August8 import predict_face_single, predict_text_full, fuse_emotions, COMMON_EMOTIONS

# Đường dẫn thư mục chứa dữ liệu RAVDESS chính diện của bạn
DATASET_PATH = r"E:\HK8\PBL4\Code\RAVDESS_Sub"

# Bảng ánh xạ nâng cao mã số RAVDESS sang nhãn hệ thống v8
RAVDESS_CODE = {
    "01": "neutral",
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "08": "surprise"
}

VIETNAMESE_TEXT_INJECTION = {
    "neutral": "Mọi chuyện vẫn bình thường, không có gì quá đặc biệt xảy ra hôm nay cả",
    "joy": "Mọi chuyện đang rất tốt đẹp, mình cảm thấy tràn đầy năng lượng và vui vẻ",
    "sadness": "Dạo này áp lực học tập quá, mình cảm thấy bế tắc và mệt mỏi vô cùng",
    "anger": "Thật sự không thể chịu đựng nổi nữa, quá rắc rối và bực mình rồi",
    "fear": "Mình lo lắng quá, không biết kỳ thi sắp tới có suôn sẻ không nữa",
    "surprise": "Oa, bất ngờ quá, mình không nghĩ là kết quả hệ thống lại có thể tốt đến như vậy"
}

def parse_emotion_from_html(html_str):
    if not html_str: return "neutral"
    if "Vui vẻ" in html_str or "Yêu thương" in html_str: return "joy"
    if "Buồn bã" in html_str: return "sadness"
    if "Tức giận" in html_str: return "anger"
    if "Lo lắng" in html_str: return "fear"
    if "Ngạc nhiên" in html_str: return "surprise"
    if "Bình thường" in html_str: return "neutral"
    return "neutral"

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 KÍCH HOẠT TIẾN TRÌNH KIỂM THỬ DUNG HỢP CHUỖI THỜI GIAN (5 FPS)")
    print("="*60)
    
    y_true = []
    y_pred_text = []
    y_pred_face = []
    y_pred_fused = []
    evaluation_logs = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for file_name in files:
            if not file_name.lower().endswith('.mp4'): continue
                
            parts = file_name.split('-')
            if len(parts) < 3: continue
            
            ground_truth = RAVDESS_CODE.get(parts[2])
            if not ground_truth: continue  
            
            video_path = os.path.join(root, file_name)
            
            try:
                # 1. XỬ LÝ KÊNH THỊ GIÁC: QUÉT CHUỖI THỜI GIAN 5 FPS
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30.0
                interval = max(1, int(fps // 5))

                all_face_probs = []
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None: break
                    if frame_idx % interval == 0:
                        probs, _ = predict_face_single(frame)
                        if probs is not None:
                            all_face_probs.append(probs)
                    frame_idx += 1
                cap.release()

                if all_face_probs:
                    face_probs = {e: 0.0 for e in COMMON_EMOTIONS}
                    for p_dict in all_face_probs:
                        for e in COMMON_EMOTIONS:
                            face_probs[e] += p_dict[e]
                    for e in COMMON_EMOTIONS:
                        face_probs[e] /= len(all_face_probs)
                    p_face = max(face_probs, key=face_probs.get)
                else:
                    p_face = "neutral"
                    face_probs = None

                # 2. INJECT KÊNH NGÔN NGỮ
                injected_text = VIETNAMESE_TEXT_INJECTION[ground_truth]
                text_probs = predict_text_full(injected_text)
                p_text = max(text_probs, key=text_probs.get)
                
                # 3. DUNG HỢP
                p_fused, final_conf, _, w_text, w_face = fuse_emotions(text_probs, face_probs)
                
                # LƯU DỮ LIỆU
                y_true.append(ground_truth)
                y_pred_text.append(p_text)
                y_pred_face.append(p_face)
                y_pred_fused.append(p_fused)
                
                evaluation_logs.append({
                    "Tên File": file_name, "Nhãn Thực Tế": ground_truth,
                    "Chỉ Dùng Text": p_text, "Chỉ Dùng Face": p_face, "Kết Quả Dung Hợp": p_fused,
                    "Trọng Số Text": round(w_text, 2), "Trọng Số Face": round(w_face, 2)
                })
                
            except Exception as e:
                print(f"❌ Lỗi file {file_name}: {e}")

    # ── 4. TRÍCH XUẤT SỐ LIỆU ĐỐI SÁNH TRÊN TERMINAL ──
    acc_text = accuracy_score(y_true, y_pred_text)
    acc_face = accuracy_score(y_true, y_pred_face)
    acc_fused = accuracy_score(y_true, y_pred_fused)
    fusion_gain = acc_fused - max(acc_text, acc_face)

    unique_labels = sorted(list(set(y_true)))

    print("\n" + "="*60)
    print("📊 BÁO CÁO HIỆU NĂNG ĐỐI SÁNH DUNG HỢP (THỜI GIAN THỰC)")
    print("="*60)
    print(f"• Accuracy đơn kênh Ngôn ngữ (Text) : {acc_text:.2%}")
    print(f"• Accuracy đơn kênh Thị giác (Face)  : {acc_face:.2%}")
    print(f"• Accuracy Hệ thống Đa phương thức   : {acc_fused:.2%}")
    print(f"➔ CHỈ SỐ FUSION GAIN THỰC TẾ         : {fusion_gain:+.2%}")
    
    pd.DataFrame(evaluation_logs).to_excel("ablation_study_report(resnettop1).xlsx", index=False)
    print("💾 Đã lưu dữ liệu Excel.")

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. XUẤT ẢNH BÁO CÁO (DASHBOARD) TỰ ĐỘNG TỪ DỮ LIỆU THỰC TẾ VỪA CHẠY
    # ═══════════════════════════════════════════════════════════════════════════
    print("🎨 Đang tổng hợp số liệu và vẽ biểu đồ Dashboard Đa phương thức...")
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))


    display_labels = [lbl.capitalize() for lbl in unique_labels]

    # [PANEL 1] Đặc trưng Tập dữ liệu (Từ dữ liệu thật y_true)
    counts_dict = Counter(y_true)
    counts = [counts_dict[e] for e in unique_labels]
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(display_labels, counts, color='#3b82f6', edgecolor='#1d4ed8', width=0.6)
    ax1.set_title("1. Dataset Characteristics & Sampling Strategy", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax1.set_ylabel("Number of Video Samples", fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.3)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.02), str(yval), ha='center', va='bottom', fontweight='bold')

    # [PANEL 2] Kết quả Ablation Study (Từ biến accuracy thật)
    configs = ['Text Only', 'Face Only\n(@ 5 FPS)', 'Multimodal\nFused']
    accuracies = [acc_text * 100, acc_face * 100, acc_fused * 100]
    
    ax2 = axes[0, 1]
    colors2 = ['#6366f1', '#f59e0b', '#10b981']
    bars2 = ax2.bar(configs, accuracies, color=colors2, width=0.5, edgecolor='#334155')
    ax2.set_title("2. Ablation Study - Accuracy Comparison", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax2.set_ylabel("Accuracy (%)", fontweight='bold')
    ax2.set_ylim(0, 115)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

    # [PANEL 3] Ma trận nhầm lẫn Face Only (Từ mảng y_pred_face thật)
    cm_face = confusion_matrix(y_true, y_pred_face, labels=unique_labels)
    ax3 = axes[1, 0]
    sns.heatmap(cm_face, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
                xticklabels=display_labels, yticklabels=display_labels, ax=ax3, annot_kws={"weight": "bold", "size": 11})
    ax3.set_title(f"3. Face Only Confusion Matrix (Accuracy: {acc_face:.2%})", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax3.set_ylabel("Actual Emotion (Ground Truth)", fontweight='bold')
    ax3.set_xlabel("Predicted Emotion by Face Model", fontweight='bold')


    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_image_path = "evaluation_metrics_dashboard(resnettop1).png"
    plt.savefig(output_image_path, dpi=200)
    plt.close()

    print(f"🎉 Xuất thành công đồ thị tổng hợp chất lượng cao tại: {os.path.abspath(output_image_path)}")
    print("="*60 + "\n")