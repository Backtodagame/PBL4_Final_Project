"""
Mã nguồn thực nghiệm tự động v8.2 — Tích hợp PhoWhisper & Dashboard 4 Panel
Sử dụng trên tập dữ liệu Demo 70 Clips (Tiếng Việt).
"""

import os
import re
import pandas as pd
import numpy as np
import cv2
import librosa
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
from moviepy import VideoFileClip

# Import thêm asr_pipe từ file hệ thống của bạn
from mental_health_ui_August10 import (
    predict_face_single, predict_text_full, fuse_emotions, 
    COMMON_EMOTIONS, asr_pipe
)

# 🛠️ SỬA LẠI ĐƯỜNG DẪN TRỎ VÀO THƯ MỤC 70 CLIP DEMO CỦA BẠN
DATASET_PATH = r"E:\HK8\PBL4\Code\Demo_70_FUll_Clips_VN"

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 KÍCH HOẠT KIỂM THỬ TOÀN TRÌNH ĐA PHƯƠNG THỨC (END-TO-END)")
    print("="*60)
    
    y_true, y_pred_text, y_pred_face, y_pred_fused = [], [], [], []
    evaluation_logs = []

    # 1. TÌM KIẾM FILE
    all_videos = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for f in files:
            if f.lower().endswith('.mp4'):
                all_videos.append(os.path.join(root, f))
                
    if len(all_videos) == 0:
        print(f"❌ CẢNH BÁO: Không tìm thấy file .mp4 nào trong {DATASET_PATH}")
        print("Vui lòng kiểm tra lại đường dẫn!")
        exit()

    print(f"📁 Tìm thấy {len(all_videos)} video. Bắt đầu xử lý...\n")

    count = 0
    for video_path in all_videos:
        file_name = os.path.basename(video_path)
        
        # 🛠️ LOGIC ĐỌC TÊN FILE MỚI: Demo_Actor_01_SADNESS.mp4
        try:
            parts = file_name.split('_')
            ground_truth = parts[3].split('.')[0].lower()
            if ground_truth not in COMMON_EMOTIONS:
                continue
        except Exception:
            continue
            
        count += 1
        print(f"[{count}/{len(all_videos)}] Đang phân tích: {file_name} (Nhãn: {ground_truth})")
        
        try:
            # ─────────────────────────────────────────────────────────
            # KÊNH 1: THỊ GIÁC (Quét 5 FPS)
            # ─────────────────────────────────────────────────────────
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            interval = max(1, int(fps // 5))
            
            all_face_probs = []
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % interval == 0:
                    probs, _ = predict_face_single(frame)
                    if probs: all_face_probs.append(probs)
                frame_idx += 1
            cap.release()

            if all_face_probs:
                face_probs = {e: sum(p[e] for p in all_face_probs)/len(all_face_probs) for e in COMMON_EMOTIONS}
                p_face = max(face_probs, key=face_probs.get)
            else:
                face_probs = None
                p_face = "neutral"

            # ─────────────────────────────────────────────────────────
            # KÊNH 2: NGÔN NGỮ (Trích Audio -> PhoWhisper -> XLM-R)
            # ─────────────────────────────────────────────────────────
            clip = VideoFileClip(video_path)
            temp_wav = "temp_eval_audio.wav"
            
            if clip.audio:
                clip.audio.write_audiofile(temp_wav, fps=16000, logger=None)
                speech_arr, _ = librosa.load(temp_wav, sr=16000)
                asr_res = asr_pipe({'array': speech_arr, 'sampling_rate': 16000})
                transcribed_text = re.sub(r'<\|[^|]+\|>', '', asr_res['text']).strip()
            else:
                transcribed_text = ""
                
            clip.close()
            if os.path.exists(temp_wav): os.remove(temp_wav)
            
            text_probs = predict_text_full(transcribed_text)
            p_text = max(text_probs, key=text_probs.get)
            
            # ─────────────────────────────────────────────────────────
            # KÊNH 3: DUNG HỢP (LATE FUSION)
            # ─────────────────────────────────────────────────────────
            p_fused, final_conf, _, w_text, w_face = fuse_emotions(text_probs, face_probs)
            
            # LƯU LOG
            y_true.append(ground_truth)
            y_pred_text.append(p_text)
            y_pred_face.append(p_face)
            y_pred_fused.append(p_fused)
            
            evaluation_logs.append({
                "Tên File": file_name, "Nhãn Thực Tế": ground_truth,
                "PhoWhisper Nhận Diện": transcribed_text,
                "Chỉ Dùng Text": p_text, "Chỉ Dùng Face": p_face, "Kết Quả Dung Hợp": p_fused,
                "Trọng Số Text": round(w_text, 2), "Trọng Số Face": round(w_face, 2)
            })
            
        except Exception as e:
            print(f"  ❌ Lỗi xử lý file {file_name}: {e}")

    # =========================================================================
    # TỔNG HỢP & VẼ DASHBOARD
    # =========================================================================
    if len(y_true) == 0:
        print("\n⚠️ Không có dữ liệu hợp lệ để đánh giá.")
        exit()

    acc_text = accuracy_score(y_true, y_pred_text)
    acc_face = accuracy_score(y_true, y_pred_face)
    acc_fused = accuracy_score(y_true, y_pred_fused)
    fusion_gain = acc_fused - max(acc_text, acc_face)

    print("\n" + "="*60)
    print("📊 BÁO CÁO HIỆU NĂNG TOÀN TRÌNH (PhoWhisper + ResNet + XLM-R)")
    print("="*60)
    print(f"• Accuracy Text (PhoWhisper + XLM-R) : {acc_text:.2%}")
    print(f"• Accuracy Face (ResNet-18 @ 5FPS)   : {acc_face:.2%}")
    print(f"• Accuracy DUNG HỢP (Hệ thống cuối)  : {acc_fused:.2%}")
    print(f"➔ CHỈ SỐ FUSION GAIN THỰC TẾ         : {fusion_gain:+.2%}")
    
    pd.DataFrame(evaluation_logs).to_excel("End_to_End_Report.xlsx", index=False)
    print("💾 Đã lưu dữ liệu chi tiết vào file: End_to_End_Report.xlsx")

    # ── VẼ DASHBOARD 4 PANEL ──
    print("🎨 Đang xuất biểu đồ Dashboard Đa phương thức...")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    unique_labels = sorted(list(set(y_true)))
    display_labels = [lbl.capitalize() for lbl in unique_labels]

    # PANEL 1: Dataset
    counts_dict = Counter(y_true)
    counts = [counts_dict[e] for e in unique_labels]
    ax1 = axes[0, 0]
    bars1 = ax1.bar(display_labels, counts, color='#3b82f6', edgecolor='#1d4ed8', width=0.6)
    ax1.set_title("1. Test Dataset Distribution", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax1.set_ylabel("Number of Videos", fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.3)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.02), str(yval), ha='center', va='bottom', fontweight='bold')

    # PANEL 2: Ablation Study
    configs = ['Text Pipeline\n(PhoWhisper+XLMR)', 'Face Pipeline\n(ResNet18 @ 5FPS)', 'Multimodal\nFused']
    accuracies = [acc_text * 100, acc_face * 100, acc_fused * 100]
    ax2 = axes[0, 1]
    colors2 = ['#6366f1', '#f59e0b', '#10b981']
    bars2 = ax2.bar(configs, accuracies, color=colors2, width=0.5, edgecolor='#334155')
    ax2.set_title("2. End-to-End Accuracy Comparison", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax2.set_ylabel("Accuracy (%)", fontweight='bold')
    ax2.set_ylim(0, 115)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

    # PANEL 3: Face CM
    cm_face = confusion_matrix(y_true, y_pred_face, labels=unique_labels)
    ax3 = axes[1, 0]
    sns.heatmap(cm_face, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
                xticklabels=display_labels, yticklabels=display_labels, ax=ax3, annot_kws={"weight": "bold", "size": 11})
    ax3.set_title(f"3. Face Only Confusion Matrix (Acc: {acc_face:.2%})", fontsize=12, fontweight='bold', color='#1e3a8a')
    ax3.set_ylabel("Ground Truth", fontweight='bold')
    ax3.set_xlabel("Predicted by Face", fontweight='bold')

    # PANEL 4: Fused CM
    cm_fused = confusion_matrix(y_true, y_pred_fused, labels=unique_labels)
    ax4 = axes[1, 1]
    sns.heatmap(cm_fused, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=display_labels, yticklabels=display_labels, ax=ax4, annot_kws={"weight": "bold", "size": 11})
    ax4.set_title(f"4. Multimodal Fused Confusion Matrix (Acc: {acc_fused:.2%})", fontsize=12, fontweight='bold', color='#065f46')
    ax4.set_ylabel("Ground Truth", fontweight='bold')
    ax4.set_xlabel("Predicted by Fused System", fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_image_path = "End_to_End_Dashboard.png"
    plt.savefig(output_image_path, dpi=200)
    plt.close()

    print(f"🎉 Xuất thành công Dashboard tại: {os.path.abspath(output_image_path)}")
    print("="*60 + "\n")