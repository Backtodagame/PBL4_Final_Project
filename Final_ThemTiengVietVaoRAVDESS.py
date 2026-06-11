import os
import math
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# ══════════════════════════════════════════════════════════════
# CẤU HÌNH ĐƯỜNG DẪN
# ══════════════════════════════════════════════════════════════
RAVDESS_DIR = r"E:\HK8\PBL4\Code\RAVDESS_Sub"          # Nguồn RAVDESS gốc
OUTPUT_DIR = r"E:\HK8\PBL4\Code\Demo_70_FUll_Clips_VN"      # Thư mục đích chứa 70 clip Demo
AUDIO_DIR = os.path.join(OUTPUT_DIR, "Temp_Audios")    # Thư mục lưu file tiếng Việt nháp

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. KỊCH BẢN 7 CẢM XÚC (THÊM DISGUST)
# ══════════════════════════════════════════════════════════════
EMOTION_MAP = {
    "01": ("neutral", "Mọi chuyện vẫn bình thường, không có gì quá đặc biệt xảy ra hôm nay cả"),
    "03": ("joy", "Mọi chuyện đang rất tốt đẹp, mình cảm thấy tràn đầy năng lượng và vui vẻ"),
    "04": ("sadness", "Dạo này áp lực học tập quá, mình cảm thấy bế tắc và mệt mỏi vô cùng"),
    "05": ("anger", "Thật sự không thể chịu đựng nổi nữa, quá rắc rối và bực mình rồi"),
    "06": ("fear", "Mình lo lắng quá, không biết kỳ thi sắp tới có suôn sẻ không nữa"),
    "07": ("disgust", "Thật ghê tởm, mình không thể chịu nổi thứ mùi vị kinh khủng này nữa"),
    "08": ("surprise", "Oa, bất ngờ quá, mình không nghĩ là kết quả hệ thống lại có thể tốt đến như vậy")
}

# ══════════════════════════════════════════════════════════════
# 2. TỰ ĐỘNG SINH ÂM THANH TIẾNG VIỆT QUA GOOGLE TTS
# ══════════════════════════════════════════════════════════════
print("🎙️ BƯỚC 1: Đang tổng hợp 7 giọng đọc tiếng Việt chuẩn...")
audio_clips = {}
for code, (emo_name, text) in EMOTION_MAP.items():
    audio_path = os.path.join(AUDIO_DIR, f"{code}_{emo_name}.mp3")
    if not os.path.exists(audio_path):
        tts = gTTS(text=text, lang='vi', slow=False)
        tts.save(audio_path)
    audio_clips[code] = audio_path
print("✅ Hoàn tất tạo Audio!")

# ══════════════════════════════════════════════════════════════
# 3. QUÉT DỮ LIỆU & GHÉP NỐI VIDEO (DOUBLE CLIP THỦ CÔNG)
# ══════════════════════════════════════════════════════════════
print("\n🎬 BƯỚC 2: Bắt đầu tiến trình Render 70 Video Demo...")

clip_count = 0
for actor_id in range(1, 11):
    actor_folder = f"Actor_{actor_id:02d}"
    actor_path = os.path.join(RAVDESS_DIR, actor_folder)
    
    if not os.path.isdir(actor_path):
        continue
        
    actor_out_dir = os.path.join(OUTPUT_DIR, actor_folder)
    os.makedirs(actor_out_dir, exist_ok=True)
    
    files = [f for f in os.listdir(actor_path) if f.endswith('.mp4')]
    
    for code, (emo_name, _) in EMOTION_MAP.items():
        target_file = next((f for f in files if f.split('-')[2] == code), None)
        
        if target_file:
            video_in_path = os.path.join(actor_path, target_file)
            audio_in_path = audio_clips[code]
            
            out_filename = f"Demo_{actor_folder}_{emo_name.upper()}.mp4"
            out_path = os.path.join(actor_out_dir, out_filename)
            
            if os.path.exists(out_path):
                clip_count += 1
                continue
                
            try:
                # Load Video & Audio
                video = VideoFileClip(video_in_path)
                audio = AudioFileClip(audio_in_path)
                
                # ──────────────────────────────────────────────────
                # LOGIC MỚI: GIỮ TRỌN VẸN KHUNG HÌNH (KHÔNG CẮT)
                # ──────────────────────────────────────────────────
                if audio.duration > video.duration:
                    # Tính số lần cần lặp để video đủ bao phủ audio
                    loops_needed = math.ceil(audio.duration / video.duration)
                    
                    # Nhân bản video thành 1 mảng
                    clips_array = [video] * loops_needed
                    
                    # Nối chúng lại với nhau (tổng thời gian video lúc này sẽ >= audio)
                    final_video = concatenate_videoclips(clips_array)
                else:
                    # Nếu video gốc đã dài hơn audio thì giữ nguyên không lặp
                    final_video = video
                
                # Lồng tiếng: Bản 2.0 bắt buộc dùng with_audio. 
                # (Phần video thừa ra ở cuối sẽ tự động im lặng - rất tự nhiên)
                final_video = final_video.with_audio(audio)
                
                final_video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
                
                # Giải phóng RAM
                video.close()
                audio.close()
                final_video.close()
                
                clip_count += 1
                print(f"  [{clip_count}/70] Đã render: {out_filename}")
                
            except Exception as e:
                print(f"  ❌ Lỗi render file {target_file}: {e}")

print(f"\n🎉 THÀNH CÔNG RỰC RỠ! {clip_count} Video Demo hoàn hảo đã nằm trong: {OUTPUT_DIR}")