"""
Mental Health Monitor — Gradio UI v7
Tích hợp: Text Emotion + Face Emotion + Speech-to-Text (PhoWhisper)

THAY ĐỔI SO VỚI v6:
  1. Màn hình chào đẹp: nhập tên trước khi vào app
  2. Tab lịch sử: donut chart cảm xúc + line chart sức khỏe 7 ngày
     (bỏ bảng 14 ngày rườm rà)
  3. Lịch sử dùng chung: tất cả máy kết nối đều ghi/đọc cùng 1 file JSON
     trên máy chủ, phân biệt nhau bằng tên người dùng
  4. Dùng cloudflared thay ngrok để share ra internet

CÁCH CHẠY:
  Terminal 1: python mental_health_ui_v7.py
  Terminal 2: .\\cloudflared-windows-amd64.exe tunnel --url http://localhost:7860
  → Gửi link https://xxxx.trycloudflare.com cho người khác
"""

import torch
import torch.nn as nn
import cv2
import json
import os
import re
import random
import numpy as np
from datetime import datetime, date
from collections import Counter
from torchvision import models, transforms
from transformers import (
    XLMRobertaForSequenceClassification, AutoTokenizer, pipeline
)
import gradio as gr
import threading, base64, tempfile

# ══════════════════════════════════════════════════════════════
#  CẤU HÌNH — chỉnh đường dẫn tại đây
# ══════════════════════════════════════════════════════════════

TEXT_MODEL_PATH   = r"E:\HK8\PBL4\Code\emotion_model_v5.3"
FACE_MODEL_PATH   = r"E:\HK8\PBL4\Code\FolderGithub\resnetL3.pth"
SPEECH_MODEL_PATH = r"E:\HK8\PBL4\Code\speak_model\phowhisper_small_vietsuperspeech"
HISTORY_FILE      = r"E:\HK8\PBL4\Code\FolderGithub\mental_health_history.json"

ALERT_CONSECUTIVE = 3
ALERT_CONF        = 0.65

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
COMMON_EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']
TEXT_EMOTIONS   = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']
FACE_EMOTIONS   = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
FACE_TO_COMMON  = {
    'happiness': 'joy', 'sadness': 'sadness', 'anger': 'anger',
    'fear': 'fear', 'surprise': 'surprise', 'neutral': 'neutral', 'disgust': 'anger'
}
EMOTION_VI = {
    'sadness': '😢 Buồn bã', 'joy': '😊 Vui vẻ', 'love': '🥰 Yêu thương',
    'anger': '😠 Tức giận', 'fear': '😰 Lo lắng', 'surprise': '😲 Ngạc nhiên',
    'neutral': '😐 Bình thường'
}
EMOTION_COLOR = {
    'sadness': '#6B9BD2', 'joy': '#F5A623', 'love': '#E85D9A',
    'anger': '#E84040', 'fear': '#9B59B6', 'surprise': '#27AE60', 'neutral': '#7F8C8D'
}
NEGATIVE_EMOTIONS = {'sadness', 'anger', 'fear'}
POSITIVE_EMOTIONS = {'joy', 'love', 'surprise'}

MUSIC = {
    'sadness': [
        {'title': 'Tâm Sự Tuổi 30 - Trịnh Thăng Bình', 'url': 'https://youtu.be/kV3famkRaA4'},
        {'title': 'Chỉ Là Không Cùng Nhau - Tăng Phúc', 'url': 'https://youtu.be/xBjJFoDK1Zw'},
        {'title': 'Someone Like You - Adele', 'url': 'https://youtu.be/hLQl3WQQoQ0'},
        {'title': 'Fix You - Coldplay', 'url': 'https://youtu.be/k4V3Mo61fJM'},
        {'title': 'Sau Tất Cả - Erik', 'url': 'https://youtu.be/wHF3Jv6Gk2o'},
    ],
    'joy': [
        {'title': 'Hãy Trao Cho Anh - Sơn Tùng M-TP', 'url': 'https://youtu.be/knW7-x7Y7RE'},
        {'title': 'Happy - Pharrell Williams', 'url': 'https://youtu.be/ZbZSe6N_BXs'},
        {'title': 'Good as Hell - Lizzo', 'url': 'https://youtu.be/SmbmeOgWsqE'},
        {'title': 'Đây là một bài hát vui - Jun Phạm', 'url': 'https://youtu.be/lZ8Ru-hAg9s'},
        {'title': 'Cứ vui lên - Mỹ Tâm', 'url': 'https://youtu.be/y70kmGVY2tA'},
    ],
    'love': [
        {'title': 'Yêu Được Không - Đức Phúc', 'url': 'https://youtu.be/_VGm6brq1aI'},
        {'title': 'Perfect - Ed Sheeran', 'url': 'https://youtu.be/cNGjD0VG4R8'},
        {'title': 'Yêu 5 - Rymastic', 'url': 'https://youtu.be/QFQdIvKSQ2Q'},
        {'title': 'Tháng 4 Là Lời Nói Dối - Hà Anh Tuấn', 'url': 'https://youtu.be/UCXao7aTDQM'},
        {'title': 'A Thousand Years - C.Perri', 'url': 'https://youtu.be/rtOvBOTyX00'},
    ],
    'anger': [
        {'title': 'Hít vào thở ra - Min x Hieuthuhai', 'url': 'https://youtu.be/Q3xlEH3_HGA'},
        {'title': 'Những ngày trời bao la - Bùi Công Nam', 'url': 'https://youtu.be/q0tvU2MFyVA'},
        {'title': 'Một Ngày Chẳng Nắng - Pháo', 'url': 'https://youtu.be/ABuY4KUUVcI'},
        {'title': 'Mùa hè tuyệt vời - Đức Phúc x Tăng Duy Tân', 'url': 'https://youtu.be/2YoIKPOUwIM'},
        {'title': 'Cứ Chill Thôi - Chillies', 'url': 'https://youtu.be/LZN4I3K8SC0'},
    ],
    'fear': [
        {'title': 'Xe đạp - Thuỳ Chi', 'url': 'https://youtu.be/6KJrNWC0tfw'},
        {'title': 'Breathe (2AM) - Anna Nalick', 'url': 'https://youtu.be/FcvXr-9XtgA'},
        {'title': 'Mọi chuyện rồi cũng sẽ qua - duongw', 'url': 'https://youtu.be/7ssyAFpQqCg'},
        {'title': 'Shape of You - Ed Sheeran', 'url': 'https://youtu.be/JGwWNGJdvx8'},
        {'title': 'Sẽ Ổn Thôi - Khải', 'url': 'https://youtu.be/TyZasCMDf5M'},
    ],
    'surprise': [
        {'title': 'Shake It Off - Taylor Swift', 'url': 'https://youtu.be/nfWlot6h_JM'},
        {'title': 'Sáng mắt chưa? - Trúc Nhân', 'url': 'https://youtu.be/rDhx4ejrPPA'},
        {'title': 'Thật bất ngờ - Trúc Nhân', 'url': 'https://youtu.be/YUAmi7Q2F0Y'},
        {'title': 'Vũ điệu cồng chiêng - Tóc Tiên', 'url': 'https://youtu.be/Rz4FbACtfd0'},
        {'title': 'GANGNAM STYLE - PSY', 'url': 'https://youtu.be/9bZkp7q19f0'},
    ],
    'neutral': [
        {'title': 'Không Cảm Xúc - Hồ Quang Hiếu', 'url': 'https://youtu.be/YZIjQDZl6Ko'},
        {'title': 'Weightless - Marconi Union', 'url': 'https://youtu.be/UfcAVejslrU'},
        {'title': 'Vì tôi còn sống - Tiên Tiên', 'url': 'https://youtu.be/Of-UkRiRWeo'},
        {'title': 'Việt Nam những chuyến đi - Vicky Nhung', 'url': 'https://youtu.be/46EjkkDo00g'},
        {'title': 'Trốn Tìm - Đen Vâu', 'url': 'https://youtu.be/Ws-QlpSltr8'},
    ],
}

BREATHING = {
    'sadness':  {'name': 'Thở 4-7-8 (Thư giãn sâu)',
                 'steps': ['Ngồi thẳng lưng, thả lỏng vai', 'Hít vào qua mũi 4 giây',
                           'Nín thở 7 giây', 'Thở ra qua miệng 8 giây', 'Lặp lại 4 lần, 2 lần/ngày']},
    'anger':    {'name': 'Thở hộp (Box Breathing)',
                 'steps': ['Ngồi thoải mái, thở ra hết', 'Hít vào 4 giây', 'Nín thở 4 giây',
                           'Thở ra 4 giây', 'Nín thở 4 giây', 'Lặp lại 4-6 lần']},
    'fear':     {'name': 'Thở 5-5-5 (Chống lo âu)',
                 'steps': ['Nhắm mắt, tập trung hơi thở', 'Hít vào 5 giây', 'Giữ 5 giây',
                           'Thở ra 5 giây', 'Nhủ thầm "Tôi an toàn"', 'Lặp lại 6-10 lần']},
    'joy':      {'name': 'Thiền biết ơn',
                 'steps': ['Nhắm mắt, hít thở tự nhiên', 'Nghĩ 3 điều tốt hôm nay',
                           'Cảm nhận sự biết ơn', 'Hít vào khi nghĩ điều tốt',
                           'Thở ra với nụ cười nhẹ', 'Giữ 5 phút']},
    'love':     {'name': 'Thiền từ bi',
                 'steps': ['Ngồi thoải mái, nhắm mắt', 'Đặt tay lên ngực',
                           'Hít vào: "Tôi xứng đáng yêu thương"',
                           'Thở ra: gửi yêu thương đến người thân',
                           'Lặp lại với từng người', 'Kết thúc: gửi đến tất cả']},
    'surprise': {'name': 'Thở cân bằng',
                 'steps': ['Ngồi yên, hít thở bình thường', 'Hít vào đều 4 giây',
                           'Thở ra đều 4 giây', 'Tập trung không khí vào/ra',
                           'Dần tăng lên 6-8 giây', 'Thực hiện 5-10 phút']},
    'neutral':  {'name': 'Thiền chánh niệm',
                 'steps': ['Ngồi/nằm thoải mái', 'Nhắm mắt, tập trung hơi thở',
                           'Quan sát bụng phồng/xẹp', 'Nếu tâm lang thang, đưa về hơi thở',
                           'Không phán xét, chỉ quan sát', '10-15 phút/ngày']},
}

ACTIVITIES = {
    'sadness':  ['🚶 Đi bộ 20 phút ngoài trời', '📔 Viết nhật ký cảm xúc',
                 '☎️ Gọi người thân tin tưởng', '🎨 Vẽ/tô màu tự do', '🛁 Tắm nước ấm thư giãn'],
    'joy':      ['💃 Nhảy theo nhạc yêu thích', '📸 Chụp ảnh kỷ niệm',
                 '🤝 Chia sẻ niềm vui với người thân',
                 '🎯 Làm việc quan trọng nhờ năng lượng tích cực', '🍳 Nấu món yêu thích'],
    'love':     ['💌 Nhắn tin cho người bạn yêu', '📷 Xem lại ảnh kỷ niệm',
                 '🎁 Chuẩn bị bất ngờ nhỏ', '🌸 Mua hoa thể hiện tình cảm', '📞 Video call nếu xa nhau'],
    'anger':    ['🥊 Chạy bộ hoặc tập gym', '✍️ Viết ra rồi xé tờ giấy đó đi',
                 '🧊 Rửa mặt nước lạnh', '🚪 Đi bộ nhanh 10 phút',
                 '⏰ Đếm ngược từ 10 trước khi phản ứng'],
    'fear':     ['📝 Viết ra nỗi lo và phân tích thực tế', '👥 Tâm sự với người tin tưởng',
                 '🎯 Chia nhỏ vấn đề thành bước nhỏ', '📚 Đọc sách/xem phim hài',
                 '🌙 Chuẩn bị kỹ cho việc sắp xảy ra'],
    'surprise': ['📖 Xử lý thông tin từ từ, đừng vội', '🫁 Hít sâu 3 lần trước khi quyết định',
                 '📞 Hỏi ý kiến người thân', '✏️ Viết ưu/nhược điểm tình huống mới',
                 '🍵 Pha trà ngồi suy nghĩ bình tĩnh'],
    'neutral':  ['🎯 Đặt mục tiêu nhỏ cho hôm nay', '📚 Đọc sách hoặc nghe podcast',
                 '🌿 Dọn dẹp không gian sống', '🏃 Vận động nhẹ 15-20 phút',
                 '📱 Hỏi thăm người lâu chưa liên lạc'],
}

# ══════════════════════════════════════════════════════════════
#  MODEL — PHẢI DÙNG ĐÚNG PureResNet18 (khớp file .pth đã train)
# ══════════════════════════════════════════════════════════════
class PureResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(PureResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Text model
print('📥 Loading Text Model...')
config_path = f'{TEXT_MODEL_PATH}/config.json'
with open(config_path, encoding='utf-8') as f:
    cfg = json.load(f)
if cfg.get('model_type') != 'xlm-roberta':
    cfg.update({'model_type': 'xlm-roberta', 'vocab_size': 250002,
                'type_vocab_size': 1, 'max_position_embeddings': 514})
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = XLMRobertaForSequenceClassification.from_pretrained(
    TEXT_MODEL_PATH, ignore_mismatched_sizes=True,
    torch_dtype=torch.float16 # BỔ SUNG DÒNG NÀY ĐỂ ÉP CÂN
    ).to(device).eval()
print('✅ Text model loaded!')

# Face model
print('📥 Loading Face Model...')
face_model = PureResNet18(num_classes=7).to(device)
face_available = False
if os.path.exists(FACE_MODEL_PATH):
    try:
        ckpt = torch.load(FACE_MODEL_PATH, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # 🛠️ CƠ CHẾ FIX LỆCH KEY: Tự động chèn 'resnet.' nếu thiếu
        first_key = list(state_dict.keys())[0]
        if not first_key.startswith('resnet.'):
            state_dict = {f'resnet.{k}': v for k, v in state_dict.items()}
            
        face_model.load_state_dict(state_dict)
        face_model.eval()
        face_available = True
        print('✅ Face model loaded!')
    except Exception as e:
        print(f'⚠️  Lỗi load face model: {e}')
else:
    print(f'⚠️  Không tìm thấy: {FACE_MODEL_PATH}')

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Speech model
print('📥 Loading Speech Model (PhoWhisper)...')
speech_available = False
asr_pipe = None
if os.path.exists(SPEECH_MODEL_PATH):
    try:
        asr_pipe = pipeline(
            'automatic-speech-recognition',
            model=SPEECH_MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16, # BỔ SUNG DÒNG NÀY ĐỂ ÉP CÂN 
            generate_kwargs={
                "language": "vi", "task": "transcribe",
                "no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "num_beams": 5,
            }
        )
        speech_available = True
        print('✅ Speech model loaded!')
    except Exception as e:
        print(f'⚠️  Lỗi load speech model: {e}')
else:
    print(f'⚠️  Không tìm thấy: {SPEECH_MODEL_PATH}')


import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 Đã dọn dẹp rác VRAM thành công!")
    
# ══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════
def predict_text_full(text):
    inputs = text_tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        probs = torch.softmax(text_model(**inputs).logits, dim=-1)[0].cpu().tolist()
    return {e: p for e, p in zip(TEXT_EMOTIONS, probs)}


def predict_face_single(frame):
    """Nhận BGR frame, trả về (common_probs, bbox) hoặc (None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    inp = face_transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = face_model(inp)
        probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
    common = {e: 0.0 for e in COMMON_EMOTIONS}
    for fl, p in zip(FACE_EMOTIONS, probs):
        common[FACE_TO_COMMON[fl]] += p
    return common, (x, y, w, h)


def resample_audio(arr, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return arr
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(arr, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        n = int(len(arr) / orig_sr * target_sr)
        return np.interp(np.linspace(0, len(arr)-1, n), np.arange(len(arr)), arr).astype(np.float32)


def transcribe_from_gradio_audio(audio):
    if not speech_available or asr_pipe is None:
        return "⚠️ Speech model chưa load"
    if audio is None:
        return "⚠️ Chưa có audio — hãy ghi âm trước rồi nhấn Phân tích"
    try:
        sr, arr = audio
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if len(arr) < sr * 0.5:
            return "⚠️ Ghi âm quá ngắn — hãy nói ít nhất 1-2 giây rồi dừng"
        if np.abs(arr).max() < 1e-4:
            return "⚠️ Không nghe thấy gì — kiểm tra lại mic và thử lại"
        if np.abs(arr).max() > 1.0:
            arr = arr / 32768.0
        if sr != 16000:
            arr = resample_audio(arr, orig_sr=sr, target_sr=16000)
        result = asr_pipe({'array': arr, 'sampling_rate': 16000})
        text = re.sub(r'<\|[^|]+\|>', '', result['text']).strip()
        if not text:
            return "⚠️ Không nhận dạng được — thử nói to hơn hoặc gần mic hơn"
        return text
    except Exception as e:
        return f"Lỗi nhận dạng: {e}"


def fuse_emotions(text_probs, face_probs=None):
    text_conf = max(text_probs.values())
    face_conf = max(face_probs.values()) if face_probs else 0.0
    if face_probs and (text_conf + face_conf) > 0:
        w_text = text_conf / (text_conf + face_conf)
        w_face = face_conf / (text_conf + face_conf)
    else:
        w_text, w_face = 1.0, 0.0
    fused = {e: w_text * text_probs.get(e, 0.0) + w_face * (face_probs.get(e, 0.0) if face_probs else 0.0)
             for e in COMMON_EMOTIONS}
    best = max(fused, key=fused.get)
    return best, fused[best], fused, w_text, w_face


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def purge_old_data(history, keep_days=7):
    from datetime import timedelta
    cutoff = str(date.today() - timedelta(days=keep_days))
    for user in history:
        old_dates = [d for d in list(history[user].keys()) if d < cutoff]
        for d in old_dates:
            del history[user][d]
    return history


def save_history(history):
    history = purge_old_data(history)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def draw_face_label(frame_bgr, bbox, text_vi):
    from PIL import Image as PILImage, ImageDraw, ImageFont
    x, y, w, h = bbox
    img_pil = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    emotion_key = next((k for k, v in EMOTION_VI.items() if v in text_vi or text_vi in v), None)
    col_hex = EMOTION_COLOR.get(emotion_key, '#FFFFFF')
    col_rgb = tuple(int(col_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    draw.rectangle([(x, y), (x+w, y+h)], outline=col_rgb, width=2)
    font = None
    for fp in [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
               r"C:\Windows\Fonts\tahoma.ttf"]:
        if os.path.exists(fp):
            try:
                from PIL import ImageFont
                font = ImageFont.truetype(fp, 36)
                break
            except Exception:
                continue
    if font is None:
        from PIL import ImageFont
        font = ImageFont.load_default()
    tb = draw.textbbox((x, max(y - 44, 0)), text_vi, font=font)
    draw.rectangle(tb, fill=col_rgb)
    draw.text((x, max(y - 44, 0)), text_vi, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_health_score(records):
    if not records:
        return 50
    neg = sum(1 for r in records if r['final_emotion'] in NEGATIVE_EMOTIONS) / len(records)
    pos = sum(1 for r in records if r['final_emotion'] in POSITIVE_EMOTIONS) / len(records)
    return max(0, min(100, int((1 - neg) * 70 + pos * 30)))


def make_bar_html(probs):
    bars = ""
    for e in COMMON_EMOTIONS:
        p = probs.get(e, 0.0)
        pct = int(p * 100)
        col = EMOTION_COLOR[e]
        vi = EMOTION_VI[e]
        bars += (
            f'<div style="display:flex;align-items:center;margin:4px 0;gap:8px">'
            f'<span style="width:120px;font-size:13px;color:#ddd">{vi}</span>'
            f'<div style="flex:1;background:#2a2a3e;border-radius:4px;height:18px;overflow:hidden">'
            f'<div style="width:{pct}%;background:{col};height:100%;border-radius:4px"></div></div>'
            f'<span style="width:38px;text-align:right;font-size:13px;color:{col};font-weight:600">{pct}%</span></div>'
        )
    return f'<div style="padding:8px 0">{bars}</div>'


def make_suggestion_html(emotion):
    songs = random.sample(MUSIC[emotion], 2)
    b = BREATHING[emotion]
    acts = random.sample(ACTIVITIES[emotion], 3)
    col = EMOTION_COLOR[emotion]
    song_html = "".join(
        f'<a href="{s["url"]}" target="_blank" style="display:block;color:{col};'
        f'text-decoration:none;margin:4px 0;font-size:13px">🎵 {s["title"]}</a>'
        for s in songs
    )
    steps_html = "".join(f'<div style="font-size:13px;color:#ccc;margin:3px 0">• {st}</div>' for st in b['steps'])
    acts_html = "".join(f'<div style="font-size:13px;color:#ccc;margin:3px 0">{a}</div>' for a in acts)
    return (
        f'<div style="background:#1a1a2e;border-radius:12px;padding:16px;border-left:4px solid {col}">'
        f'<div style="margin-bottom:14px"><div style="color:{col};font-weight:700;font-size:14px;margin-bottom:6px">🎵 NHẠC PHÙ HỢP</div>{song_html}</div>'
        f'<div style="margin-bottom:14px"><div style="color:{col};font-weight:700;font-size:14px;margin-bottom:6px">🫁 {b["name"]}</div>{steps_html}</div>'
        f'<div><div style="color:{col};font-weight:700;font-size:14px;margin-bottom:6px">🎯 HOẠT ĐỘNG GỢI Ý</div>{acts_html}</div>'
        f'</div>'
    )


def make_score_html(user_name, history):
    today = str(date.today())
    records = history.get(user_name, {}).get(today, [])
    if not records:
        return ""
    score = get_health_score(records)
    col = '#27AE60' if score >= 65 else '#F5A623' if score >= 40 else '#E84040'
    status = '🟢 Ổn định' if score >= 65 else '🟡 Cần theo dõi' if score >= 40 else '🔴 Cần hỗ trợ'
    counts = Counter(r['final_emotion'] for r in records)
    dominant = counts.most_common(1)[0][0]
    neg_pct = sum(counts.get(e, 0) for e in NEGATIVE_EMOTIONS) / len(records) * 100
    dash = int(score * 2.51)
    return (
        f'<div style="background:#1a1a2e;border-radius:12px;padding:16px;margin-top:12px">'
        f'<div style="color:#fff;font-weight:700;margin-bottom:10px;font-size:14px">📊 SỨC KHỎE HÔM NAY</div>'
        f'<div style="display:flex;align-items:center;gap:20px">'
        f'<svg width="80" height="80" viewBox="0 0 80 80">'
        f'<circle cx="40" cy="40" r="32" fill="none" stroke="#2a2a3e" stroke-width="8"/>'
        f'<circle cx="40" cy="40" r="32" fill="none" stroke="{col}" stroke-width="8"'
        f' stroke-dasharray="{dash} 251" stroke-linecap="round" transform="rotate(-90 40 40)"/>'
        f'<text x="40" y="45" text-anchor="middle" fill="{col}" font-size="16" font-weight="bold">{score}</text>'
        f'</svg>'
        f'<div><div style="color:{col};font-weight:700;font-size:15px">{status}</div>'
        f'<div style="color:#aaa;font-size:12px;margin-top:4px">Chủ đạo: {EMOTION_VI[dominant]}</div>'
        f'<div style="color:#aaa;font-size:12px">Tiêu cực: {neg_pct:.0f}% ({len(records)} lượt)</div>'
        f'</div></div></div>'
    )


def check_alerts(session_records):
    if len(session_records) < ALERT_CONSECUTIVE:
        return ""
    last = session_records[-ALERT_CONSECUTIVE:]
    if all(r['final_emotion'] in NEGATIVE_EMOTIONS and r['final_conf'] >= ALERT_CONF for r in last):
        em = last[-1]['final_emotion']
        col = EMOTION_COLOR[em]
        return (
            f'<div style="background:#2a0a0a;border:1px solid #E84040;border-radius:10px;padding:12px;margin-top:10px">'
            f'<div style="color:#E84040;font-weight:700">🚨 CẢNH BÁO</div>'
            f'<div style="color:#ccc;font-size:13px;margin-top:4px">'
            f'{EMOTION_VI[em]} liên tục {ALERT_CONSECUTIVE} lần. Hãy thử bài tập thở hoặc liên hệ người thân.</div></div>'
        )
    return ""


def get_history_html(user_name, history):
    if not user_name.strip() or user_name not in history:
        return "<div style='color:#aaa;padding:20px;text-align:center'>📭 Chưa có dữ liệu</div>"
    user_data = history[user_name]
    dates_all = sorted(user_data.keys())
    dates_7 = dates_all[-7:]
    dates_14 = dates_all[-14:]
    all_recs = [r for d in dates_all for r in user_data[d]]
    total_recs = len(all_recs)
    if not total_recs:
        return "<div style='color:#aaa;padding:20px'>Chưa có dữ liệu</div>"

    emotion_count = Counter(r['final_emotion'] for r in all_recs)
    dom_emotion = emotion_count.most_common(1)[0][0]
    neg_total = sum(emotion_count.get(e, 0) for e in NEGATIVE_EMOTIONS)
    pos_total = sum(emotion_count.get(e, 0) for e in POSITIVE_EMOTIONS)
    avg_score = sum(get_health_score(user_data[d]) for d in dates_all) / len(dates_all)
    today_recs = user_data.get(str(date.today()), [])
    today_score = get_health_score(today_recs) if today_recs else None

    def score_col(s): return '#27AE60' if s >= 65 else '#F5A623' if s >= 40 else '#E84040'
    def score_icon(s): return '🟢' if s >= 65 else '🟡' if s >= 40 else '🔴'

    avg_col = score_col(avg_score)
    stat_cards = (
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px">'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:26px;font-weight:700;color:#a78bfa">{total_recs}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Tổng lần ghi nhận</div></div>'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:26px;font-weight:700;color:#a78bfa">{len(dates_all)}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Ngày đã theo dõi</div></div>'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:22px">{EMOTION_VI[dom_emotion]}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Cảm xúc chủ đạo</div></div>'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:26px;font-weight:700;color:{avg_col}">{avg_score:.0f}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Điểm SK trung bình</div></div></div>'
    )

    today_html = ""
    if today_recs:
        ts = score_col(today_score)
        ti = score_icon(today_score)
        td = int(today_score * 2.51)
        today_html = (
            f'<div style="background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:20px;border:1px solid {ts}">'
            f'<div style="color:#fff;font-weight:700;font-size:14px;margin-bottom:12px">📅 HÔM NAY ({str(date.today())})</div>'
            f'<div style="display:flex;align-items:center;gap:20px">'
            f'<svg width="70" height="70" viewBox="0 0 80 80">'
            f'<circle cx="40" cy="40" r="32" fill="none" stroke="#2a2a3e" stroke-width="8"/>'
            f'<circle cx="40" cy="40" r="32" fill="none" stroke="{ts}" stroke-width="8"'
            f' stroke-dasharray="{td} 251" stroke-linecap="round" transform="rotate(-90 40 40)"/>'
            f'<text x="40" y="45" text-anchor="middle" fill="{ts}" font-size="16" font-weight="bold">{today_score}</text>'
            f'</svg><div style="flex:1">'
            f'<div style="color:{ts};font-weight:700;font-size:15px">{ti} '
            f'{"Ổn định" if today_score>=65 else "Cần theo dõi" if today_score>=40 else "Cần hỗ trợ"}</div>'
            f'<div style="color:#aaa;font-size:12px;margin-top:4px">{len(today_recs)} lần ghi nhận hôm nay</div>'
            f'</div></div></div>'
        )

    scores_7 = [get_health_score(user_data[d]) for d in dates_7]
    labels_7 = [d[5:] for d in dates_7]
    n = len(scores_7)
    chart_w, chart_h = 680, 180
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 40
    chart_svg = ""
    if n >= 2:
        iw = chart_w - pad_l - pad_r
        ih = chart_h - pad_t - pad_b
        def sx(i): return pad_l + i * iw / (n - 1)
        def sy(v): return pad_t + ih - (v / 100) * ih
        grid = "".join(
            f'<line x1="{pad_l}" y1="{sy(v):.1f}" x2="{chart_w-pad_r}" y2="{sy(v):.1f}"'
            f' stroke="{"#27AE60" if v==65 else "#F5A623" if v==40 else "#2a2a4e"}"'
            f' stroke-width="{"1.5" if v in (40,65) else "1"}"'
            f' stroke-dasharray="{"" if v in (40,65) else "4,3"}"/>'
            f'<text x="{pad_l-6}" y="{sy(v)+4:.1f}" text-anchor="end" fill="#555" font-size="10">{v}</text>'
            for v in [0, 25, 40, 65, 100]
        )
        pts = f"{pad_l},{pad_t+ih} " + " ".join(f"{sx(i):.1f},{sy(s):.1f}" for i,s in enumerate(scores_7)) + f" {sx(n-1):.1f},{pad_t+ih}"
        path = "M " + " L ".join(f"{sx(i):.1f},{sy(s):.1f}" for i,s in enumerate(scores_7))
        dots = "".join(
            f'<circle cx="{sx(i):.1f}" cy="{sy(s):.1f}" r="5" fill="{score_col(s)}" stroke="#0d0d1a" stroke-width="2"/>'
            f'<text x="{sx(i):.1f}" y="{sy(s)-10:.1f}" text-anchor="middle" fill="{score_col(s)}" font-size="11" font-weight="bold">{s}</text>'
            f'<text x="{sx(i):.1f}" y="{chart_h-6}" text-anchor="middle" fill="#666" font-size="10">{lbl}</text>'
            for i,(s,lbl) in enumerate(zip(scores_7,labels_7))
        )
        chart_svg = (
            f'<div style="background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:20px">'
            f'<div style="color:#fff;font-weight:700;font-size:14px;margin-bottom:12px">📈 ĐIỂM SỨC KHỎE 7 NGÀY</div>'
            f'<svg width="100%" viewBox="0 0 {chart_w} {chart_h}" style="overflow:visible">'
            f'<defs><linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#a78bfa" stop-opacity="0"/>'
            f'</linearGradient></defs>'
            f'{grid}<polygon points="{pts}" fill="url(#grad)" opacity="0.3"/>'
            f'<path d="{path}" fill="none" stroke="#a78bfa" stroke-width="2.5"/>{dots}</svg>'
            f'<div style="display:flex;gap:16px;margin-top:8px;font-size:11px">'
            f'<span style="color:#27AE60">━ ≥65 Ổn định</span>'
            f'<span style="color:#F5A623">━ 40-64 Cần theo dõi</span>'
            f'<span style="color:#E84040">━ &lt;40 Cần hỗ trợ</span></div></div>'
        )

    emotion_bars = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0">'
        f'<span style="width:130px;font-size:13px;color:#ddd;white-space:nowrap">{EMOTION_VI[em]}</span>'
        f'<div style="flex:1;background:#2a2a3e;border-radius:4px;height:20px;overflow:hidden">'
        f'<div style="width:{emotion_count.get(em,0)/total_recs*100:.0f}%;background:{EMOTION_COLOR[em]};height:100%;border-radius:4px"></div></div>'
        f'<span style="width:55px;font-size:12px;color:{EMOTION_COLOR[em]};font-weight:600;text-align:right">'
        f'{emotion_count.get(em,0)} ({emotion_count.get(em,0)/total_recs*100:.0f}%)</span></div>'
        for em in COMMON_EMOTIONS
    )
    emotion_chart = (
        f'<div style="background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:20px">'
        f'<div style="color:#fff;font-weight:700;font-size:14px;margin-bottom:12px">🎭 PHÂN BỐ CẢM XÚC ({total_recs} lần)</div>'
        f'{emotion_bars}'
        f'<div style="display:flex;gap:16px;margin-top:10px;font-size:12px">'
        f'<span style="color:#27AE60">😊 Tích cực: {pos_total} ({pos_total/total_recs*100:.0f}%)</span>'
        f'<span style="color:#E84040">😢 Tiêu cực: {neg_total} ({neg_total/total_recs*100:.0f}%)</span>'
        f'</div></div>'
    )

    rows_14 = "".join(
        f'<tr style="border-bottom:1px solid #1a1a2e">'
        f'<td style="padding:10px 12px;color:#ddd;font-size:13px">{d}</td>'
        f'<td style="padding:10px 12px"><span style="color:{score_col(get_health_score(user_data[d]))};font-weight:700">'
        f'{score_icon(get_health_score(user_data[d]))} {get_health_score(user_data[d])}</span></td>'
        f'<td style="padding:10px 12px;font-size:13px">'
        f'{EMOTION_VI[Counter(r["final_emotion"] for r in user_data[d]).most_common(1)[0][0]]}</td>'
        f'<td style="padding:10px 12px;color:#aaa;font-size:12px">{len(user_data[d])} lần</td>'
        f'</tr>'
        for d in reversed(dates_14)
    )
    table_14 = (
        f'<div style="background:#13132a;border-radius:12px;overflow:hidden;margin-bottom:20px">'
        f'<div style="color:#fff;font-weight:700;font-size:14px;padding:14px 16px;background:#1a1a2e">📋 LỊCH SỬ 14 NGÀY</div>'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr style="background:#0d0d1a">'
        f'<th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:11px">NGÀY</th>'
        f'<th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:11px">ĐIỂM SK</th>'
        f'<th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:11px">CẢM XÚC CHỦ ĐẠO</th>'
        f'<th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:11px">SỐ LẦN</th>'
        f'</tr>{rows_14}</table></div>'
    )

    detail_today = ""
    if today_recs:
        detail_rows = "".join(
            f'<tr style="border-bottom:1px solid #1a1a2e">'
            f'<td style="padding:8px 12px;color:#666;font-size:12px">{r.get("time","")}</td>'
            f'<td style="padding:8px 12px"><span style="color:{EMOTION_COLOR[r["final_emotion"]]};font-size:13px">'
            f'{EMOTION_VI[r["final_emotion"]]}</span>'
            f'{"<span style=color:#FF6B6B;font-size:11px> ⚠️ Hidden</span>" if r.get("conflict") else ""}</td>'
            f'<td style="padding:8px 12px;color:#888;font-size:11px">'
            f'{r.get("text","")[:60]}{"..." if len(r.get("text",""))>60 else ""}</td>'
            f'<td style="padding:8px 12px;color:{EMOTION_COLOR[r["final_emotion"]]};font-size:12px;font-weight:600">'
            f'{r.get("final_conf",0):.0%}</td></tr>'
            for r in reversed(today_recs[-20:])
        )
        detail_today = (
            f'<div style="background:#13132a;border-radius:12px;overflow:hidden">'
            f'<div style="color:#fff;font-weight:700;font-size:14px;padding:14px 16px;background:#1a1a2e">'
            f'🕐 CHI TIẾT HÔM NAY ({len(today_recs)} lần)</div>'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<tr style="background:#0d0d1a">'
            f'<th style="padding:8px 12px;color:#7c7caa;text-align:left;font-size:11px">GIỜ</th>'
            f'<th style="padding:8px 12px;color:#7c7caa;text-align:left;font-size:11px">CẢM XÚC</th>'
            f'<th style="padding:8px 12px;color:#7c7caa;text-align:left;font-size:11px">NỘI DUNG</th>'
            f'<th style="padding:8px 12px;color:#7c7caa;text-align:left;font-size:11px">ĐỘ TIN CẬY</th>'
            f'</tr>{detail_rows}</table></div>'
        )

    return f'<div style="padding:4px 0">{stat_cards}{today_html}{chart_svg}{emotion_chart}{table_14}{detail_today}</div>'

# ══════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════
def handle_history(user_name, _history=None):
    history = load_history()
    if not user_name.strip() or user_name not in history:
        return "<div style='color:#aaa;padding:40px;text-align:center;font-size:15px'>📭 Chưa có dữ liệu — hãy thử phân tích trước</div>"
    user_data = history[user_name]
    dates_all = sorted(user_data.keys())
    all_recs  = [r for d in dates_all for r in user_data[d]]
    total_recs = len(all_recs)
    if not total_recs:
        return "<div style='color:#aaa;padding:40px;text-align:center'>Chưa có dữ liệu</div>"

    from collections import Counter
    emotion_count = Counter(r['final_emotion'] for r in all_recs)
    dom_emotion   = emotion_count.most_common(1)[0][0]
    neg_total = sum(emotion_count.get(e,0) for e in NEGATIVE_EMOTIONS)
    pos_total = sum(emotion_count.get(e,0) for e in POSITIVE_EMOTIONS)
    avg_score = sum(get_health_score(user_data[d]) for d in dates_all) / len(dates_all)
    def score_col(s): return '#27AE60' if s>=65 else '#F5A623' if s>=40 else '#E84040'
    def score_icon(s): return '🟢' if s>=65 else '🟡' if s>=40 else '🔴'

    stat_cards = (
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px">'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:26px;font-weight:700;color:#a78bfa">{total_recs}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Tổng lần ghi nhận</div></div>'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:26px;font-weight:700;color:#a78bfa">{len(dates_all)}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Ngày theo dõi</div></div>'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a4e">'
        f'<div style="font-size:22px">{EMOTION_VI[dom_emotion]}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Cảm xúc chủ đạo</div></div>'
        f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;text-align:center;border:1px solid {score_col(avg_score)};border-width:2px">'
        f'<div style="font-size:26px;font-weight:700;color:{score_col(avg_score)}">{score_icon(avg_score)} {avg_score:.0f}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px">Điểm SK trung bình</div></div></div>'
    )

    cx, cy, r_out, r_in = 160, 160, 130, 75
    import math
    total_e = sum(emotion_count.values()) or 1
    angles, paths = [], []
    start = -math.pi / 2
    for em in COMMON_EMOTIONS:
        cnt = emotion_count.get(em, 0)
        if cnt == 0:
            continue
        sweep = 2 * math.pi * cnt / total_e
        end   = start + sweep
        lx1 = cx + r_out * math.cos(start); ly1 = cy + r_out * math.sin(start)
        lx2 = cx + r_out * math.cos(end);   ly2 = cy + r_out * math.sin(end)
        sx1 = cx + r_in  * math.cos(end);   sy1 = cy + r_in  * math.sin(end)
        sx2 = cx + r_in  * math.cos(start); sy2 = cy + r_in  * math.sin(start)
        large = 1 if sweep > math.pi else 0
        col   = EMOTION_COLOR[em]
        pct   = cnt / total_e * 100
        d_path = (f'M {lx1:.1f} {ly1:.1f} A {r_out} {r_out} 0 {large} 1 {lx2:.1f} {ly2:.1f} '
                  f'L {sx1:.1f} {sy1:.1f} A {r_in} {r_in} 0 {large} 0 {sx2:.1f} {sy2:.1f} Z')
        paths.append(f'<path d="{d_path}" fill="{col}" stroke="#0d0d1a" stroke-width="2">'
                     f'<title>{EMOTION_VI[em]}: {cnt} lần ({pct:.0f}%)</title></path>')
        angles.append((start + sweep/2, em, cnt, pct))
        start = end

    dom_vi = EMOTION_VI[dom_emotion]
    center_text = (
        f'<text x="{cx}" y="{cy-10}" text-anchor="middle" fill="#a78bfa" font-size="32">{dom_vi.split()[0]}</text>'
        f'<text x="{cx}" y="{cy+14}" text-anchor="middle" fill="#ddd" font-size="12" font-weight="600">'
        f'{" ".join(dom_vi.split()[1:])}</text>'
        f'<text x="{cx}" y="{cy+30}" text-anchor="middle" fill="#666" font-size="11">{total_recs} lần</text>'
    )
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0">'
        f'<div style="width:14px;height:14px;border-radius:3px;background:{EMOTION_COLOR[em]};flex-shrink:0"></div>'
        f'<span style="font-size:13px;color:#ddd;flex:1">{EMOTION_VI[em]}</span>'
        f'<span style="font-size:13px;color:{EMOTION_COLOR[em]};font-weight:700">{emotion_count.get(em,0)} '
        f'<span style="color:#555;font-weight:400">({emotion_count.get(em,0)/total_e*100:.0f}%)</span></span></div>'
        for em in COMMON_EMOTIONS if emotion_count.get(em, 0) > 0
    )
    donut_html = (
        f'<div style="background:#1a1a2e;border-radius:14px;padding:20px;margin-bottom:20px">'
        f'<div style="color:#fff;font-weight:700;font-size:15px;margin-bottom:16px">🎭 Biểu đồ phân bố cảm xúc</div>'
        f'<div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">'
        f'<svg width="320" height="320" viewBox="0 0 320 320">{"".join(paths)}{center_text}</svg>'
        f'<div style="flex:1;min-width:180px">{legend}</div></div>'
        f'<div style="display:flex;gap:20px;margin-top:12px;font-size:12px;border-top:1px solid #2a2a4e;padding-top:12px">'
        f'<span style="color:#27AE60">😊 Tích cực: {pos_total} ({pos_total/total_e*100:.0f}%)</span>'
        f'<span style="color:#E84040">😢 Tiêu cực: {neg_total} ({neg_total/total_e*100:.0f}%)</span>'
        f'</div></div>'
    )

    dates_7 = dates_all[-7:]
    scores_7 = [get_health_score(user_data[d]) for d in dates_7]
    labels_7 = [d[5:] for d in dates_7]
    n = len(scores_7)
    chart_svg = ""
    if n >= 2:
        cw, ch = 680, 180
        pl, pr, pt, pb = 50, 20, 20, 40
        iw = cw-pl-pr; ih = ch-pt-pb
        def sx(i): return pl + i*iw/(n-1)
        def sy(v): return pt + ih - (v/100)*ih
        grid = "".join(
            f'<line x1="{pl}" y1="{sy(v):.1f}" x2="{cw-pr}" y2="{sy(v):.1f}"'
            f' stroke="{"#27AE60" if v==65 else "#F5A623" if v==40 else "#2a2a4e"}"'
            f' stroke-width="{"1.5" if v in (40,65) else "1"}" stroke-dasharray="{"" if v in (40,65) else "4,3"}"/>'
            f'<text x="{pl-6}" y="{sy(v)+4:.1f}" text-anchor="end" fill="#555" font-size="10">{v}</text>'
            for v in [0,25,40,65,100]
        )
        pts  = f"{pl},{pt+ih} " + " ".join(f"{sx(i):.1f},{sy(s):.1f}" for i,s in enumerate(scores_7)) + f" {sx(n-1):.1f},{pt+ih}"
        path = "M " + " L ".join(f"{sx(i):.1f},{sy(s):.1f}" for i,s in enumerate(scores_7))
        dots = "".join(
            f'<circle cx="{sx(i):.1f}" cy="{sy(s):.1f}" r="5" fill="{score_col(s)}" stroke="#0d0d1a" stroke-width="2"/>'
            f'<text x="{sx(i):.1f}" y="{sy(s)-10:.1f}" text-anchor="middle" fill="{score_col(s)}" font-size="11" font-weight="bold">{s}</text>'
            f'<text x="{sx(i):.1f}" y="{ch-6}" text-anchor="middle" fill="#666" font-size="10">{lbl}</text>'
            for i,(s,lbl) in enumerate(zip(scores_7,labels_7))
        )
        chart_svg = (
            f'<div style="background:#1a1a2e;border-radius:14px;padding:20px;margin-bottom:20px">'
            f'<div style="color:#fff;font-weight:700;font-size:15px;margin-bottom:12px">📈 Điểm sức khỏe 7 ngày gần nhất</div>'
            f'<svg width="100%" viewBox="0 0 {cw} {ch}" style="overflow:visible">'
            f'<defs><linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#a78bfa" stop-opacity="0"/>'
            f'</linearGradient></defs>'
            f'{grid}<polygon points="{pts}" fill="url(#grad)" opacity="0.25"/>'
            f'<path d="{path}" fill="none" stroke="#a78bfa" stroke-width="2.5"/>{dots}</svg>'
            f'<div style="display:flex;gap:16px;margin-top:8px;font-size:11px">'
            f'<span style="color:#27AE60">━ ≥65 Ổn định</span>'
            f'<span style="color:#F5A623">━ 40-64 Cần theo dõi</span>'
            f'<span style="color:#E84040">━ &lt;40 Cần hỗ trợ</span></div></div>'
        )

    return f'<div style="padding:4px 0">{stat_cards}{donut_html}{chart_svg}</div>'

CSS = """
body, .gradio-container { background: #f3f4f6 !important; color: #1f2937 !important; font-family: 'Segoe UI', sans-serif; }
.tab-nav button { background: #ffffff !important; color: #4b5563 !important; border: none !important; padding: 10px 20px !important; font-size: 14px !important; }
.tab-nav button.selected { color: #8b5cf6 !important; border-bottom: 2px solid #8b5cf6 !important; font-weight: bold !important; }
textarea, input[type=text] { background: #ffffff !important; color: #1f2937 !important; border: 1px solid #d1d5db !important; border-radius: 8px !important; }
.gr-button { background: linear-gradient(135deg,#6366f1,#a855f7) !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.gr-button:hover { opacity: 0.9 !important; }
label { color: #4b5563 !important; font-size: 13px !important; font-weight: 600 !important; }
.gr-panel { background: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 12px !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }

.generating { border: none !important; animation: none !important; }
.generating * { border-color: transparent !important; }
.eta-bar, .progress-bar, .loader { display: none !important; }
div.generating { outline: none !important; box-shadow: none !important; }
.wrap.default.full.svelte-1ipelgc { display: none !important; }
.progress-text { display: none !important; }
footer { display: none !important; }
.svelte-1ed2p3z { display: none !important; }
span.eta-bar { display: none !important; }
.time { display: none !important; }
.meta-text { display: none !important; }
.meta-text-center { display: none !important; }
#component-0 > .wrap { display: none !important; }
.lds-ring { display: none !important; }
svg.loading { display: none !important; }
"""

with gr.Blocks(title="Mental Health Monitor", css=CSS) as app:
    session_state = gr.State([])
    history_state = gr.State(load_history())
    name_state    = gr.State("")

    # ── MÀN HÌNH CHÀO / NHẬP TÊN ─────────────────────────────────
    with gr.Column(visible=True) as welcome_screen:
        gr.HTML("""
        <div style="text-align:center;padding:60px 20px 30px">
          <div style="font-size:72px;margin-bottom:12px">🧠</div>
          <div style="font-size:30px;font-weight:700;color:#a78bfa;letter-spacing:1px;margin-bottom:8px">MENTAL HEALTH MONITOR</div>
          <div style="font-size:14px;color:#555;margin-bottom:40px">Giọng nói · Khuôn mặt · Văn bản → Phân tích cảm xúc → Gợi ý</div>
        </div>""")
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=2):
                gr.HTML('<div style="background:#1a1a2e;border-radius:16px;padding:32px;border:1px solid #2a2a4e">'
                        '<div style="color:#a78bfa;font-size:16px;font-weight:600;margin-bottom:16px;text-align:center">👤 Bạn tên là gì?</div>')
                welcome_name = gr.Textbox(label="", placeholder="Nhập tên của bạn...", max_lines=1)
                welcome_btn  = gr.Button("▶  Bắt đầu", variant="primary", size="lg")
                welcome_err  = gr.HTML("")
                gr.HTML('</div>')
            with gr.Column(scale=1): pass

    # ── APP CHÍNH ──────────────────────────────────────────────────
    with gr.Column(visible=False) as main_app:
        gr.HTML("""
        <div style="text-align:center;padding:18px 0 10px;background:linear-gradient(135deg,#0d0d1a,#1a1a2e)">
          <div style="font-size:28px;font-weight:700;color:#a78bfa;letter-spacing:1px">🧠 MENTAL HEALTH MONITOR</div>
        </div>""")
        user_greeting = gr.HTML("")
        user_name_input = gr.Textbox(visible=False)

    def do_start(name):
        name = name.strip()
        if not name:
            return (gr.update(), gr.update(),
                    '<div style="color:#E84040;text-align:center;margin-top:8px">⚠️ Vui lòng nhập tên trước khi bắt đầu</div>',
                    gr.update(), gr.update(), gr.update())
        greeting = (f'<div style="text-align:right;padding:6px 16px;color:#a78bfa;font-size:14px">'
                    f'Xin chào, <b>{name}</b> 👋</div>')
        return (gr.update(visible=False), gr.update(visible=True),
                "", greeting, gr.update(value=name), gr.update(value=name))

    welcome_btn.click(
        fn=do_start,
        inputs=[welcome_name],
        outputs=[welcome_screen, main_app, welcome_err, user_greeting, user_name_input, welcome_name],
        show_progress=False
    )

    with main_app:
      with gr.Tabs():

        # ── TAB 1: GIỌNG NÓI ─────────────────────────────────────
        with gr.Tab("🎙️ Phân tích giọng nói"):
            gr.HTML(
                f'<div style="background:#1a1a2e;border-radius:10px;padding:12px 16px;margin:8px 0">'
                f'<div style="color:{"#27AE60" if speech_available else "#E84040"};font-size:13px;margin-bottom:4px">'
                f'{"✅ PhoWhisper sẵn sàng" if speech_available else "⚠️ Speech model chưa load"}</div>'
                f'<div style="color:#8888aa;font-size:12px">'
                f'Nhấn 🔴 để bắt đầu ghi âm, nhấn lại ⏹ để dừng, rồi nhấn <b style="color:#a78bfa">Phân tích</b></div></div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:14px;margin-bottom:8px">🎙️ Ghi âm (từ trình duyệt của bạn)</div>')
                    s_audio_mic = gr.Audio(
                        label="Nhấn 🔴 để ghi âm",
                        sources=["microphone"],
                        type="numpy",
                    )
                    gr.HTML('<div style="color:#555;font-size:12px;text-align:center;margin:6px 0">── hoặc upload file WAV/MP3 ──</div>')
                    s_audio_upload = gr.Audio(
                        label="📁 Upload file audio",
                        sources=["upload"],
                        type="numpy",
                    )
                    s_analyze_btn = gr.Button("🔍 Phân tích giọng nói", variant="primary")
                    gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:14px;margin:12px 0 6px">📝 Văn bản nhận dạng được</div>')
                    s_transcribed = gr.Textbox(
                        label="", interactive=False, lines=3,
                        placeholder="Văn bản sẽ hiện ở đây sau khi nhấn Phân tích...")

                with gr.Column(scale=3):
                    gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:14px;margin-bottom:8px">🧠 Kết quả cảm xúc</div>')
                    s_emotion_label = gr.HTML()
                    s_bars          = gr.HTML()

            gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:14px;margin:14px 0 8px">💡 Gợi ý cho bạn</div>')
            s_suggestion = gr.HTML()
            s_score_out  = gr.HTML()

            def handle_speech_tab(mic_audio, upload_audio, user_name, session_records, history):
                print(f'[ENTRY speech] user={user_name}, mic={mic_audio is not None}')
                audio = mic_audio if mic_audio is not None else upload_audio
                if audio is None:
                    return "⚠️ Chưa có audio — hãy ghi âm hoặc upload file", "", "", "", "", session_records, history

                text = transcribe_from_gradio_audio(audio)
                if not text or text.startswith("⚠️") or text.startswith("Lỗi"):
                    return text, "", "", "", "", session_records, history

                if not user_name.strip():
                    user_name = "Người dùng"

                text_probs = predict_text_full(text)
                text_top   = max(text_probs, key=text_probs.get)
                final_emotion, final_conf, _, w_text, w_face = fuse_emotions(text_probs)
                col = EMOTION_COLOR[final_emotion]
                vi  = EMOTION_VI[final_emotion]

                record = {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'date': str(date.today()),
                    'text': text,
                    'text_emotion': text_top,
                    'text_conf': text_probs[text_top],
                    'face_emotion': None,
                    'face_conf': 0.0,
                    'final_emotion': final_emotion,
                    'final_conf': final_conf,
                    'w_text': w_text, 'w_face': w_face,
                    'conflict': False,
                }
                session_records = session_records + [record]
                history = load_history()
                history.setdefault(user_name, {}).setdefault(str(date.today()), []).append(record)
                try:
                    save_history(history)
                    print(f"[SAVED] user={user_name}, date={str(date.today())}")
                except Exception as e:
                    print(f"[SAVE ERROR] {e}")

                label = (
                    f'<div style="text-align:center;padding:20px;background:#1a1a2e;border-radius:12px;border:2px solid {col}">'
                    f'<div style="font-size:42px;margin-bottom:8px">{vi.split()[0]}</div>'
                    f'<div style="font-size:20px;font-weight:700;color:{col}">{" ".join(vi.split()[1:])}</div>'
                    f'<div style="font-size:14px;color:#aaa;margin-top:4px">Độ tin cậy: {final_conf:.0%}</div>'
                    f'</div>'
                )
                bars       = make_bar_html(text_probs)
                suggestion = make_suggestion_html(final_emotion)
                score_html = make_score_html(user_name, history) + check_alerts(session_records)
                return text, label, bars, suggestion, score_html, session_records, history

            s_analyze_btn.click(
                fn=handle_speech_tab,
                inputs=[s_audio_mic, s_audio_upload, user_name_input, session_state, history_state],
                outputs=[s_transcribed, s_emotion_label, s_bars, s_suggestion, s_score_out, session_state, history_state]
            )

        # ── TAB 2: KHUÔN MẶT ─────────────────────────────────────
        with gr.Tab("📷 Khuôn mặt"):
            gr.HTML(
                f'<div style="background:#1a1a2e;border-radius:10px;padding:12px 16px;margin:8px 0">'
                f'<div style="color:{"#27AE60" if face_available else "#E84040"};font-size:13px;margin-bottom:4px">'
                f'{"✅ Face model sẵn sàng" if face_available else "⚠️ Face model chưa load"}</div>'
                f'<div style="color:#8888aa;font-size:12px">'
                f'Chụp ảnh từ camera trình duyệt hoặc upload ảnh → nhấn Phân tích</div></div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    webcam_input = gr.Image(
                        label="📷 Camera / Upload ảnh",
                        sources=["webcam", "upload"],
                        type="numpy",
                    )
                    webcam_btn = gr.Button("🔍 Phân tích khuôn mặt", variant="primary")
                with gr.Column(scale=3):
                    webcam_out   = gr.Image(label="📸 Kết quả")
                    webcam_label = gr.HTML()
                    webcam_bars  = gr.HTML()
            webcam_suggestion = gr.HTML()

            def handle_webcam_tab(image, user_name, session_records):
                print(f'[ENTRY webcam] user={user_name}')
                if image is None:
                    return None, "<div style='color:#F5A623;padding:10px;background:#1a1a2e;border-radius:8px'>⚠️ Chưa có ảnh</div>", "", "", session_records
                if not face_available:
                    return None, "<div style='color:#E84040;padding:10px'>⚠️ Face model chưa load</div>", "", "", session_records
                try:
                    img_arr = np.array(image)
                    if img_arr.ndim == 2:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                    elif img_arr.shape[2] == 4:
                        img_arr = img_arr[:, :, :3]
                    frame = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                    probs, bbox = predict_face_single(frame)
                    if probs is None:
                        return img_arr, (
                            "<div style='color:#aaa;padding:12px;background:#1a1a2e;border-radius:10px'>"
                            "😶 Không phát hiện khuôn mặt<br>"
                            "<small style='color:#666'>Thử chụp gần hơn, đủ ánh sáng, nhìn thẳng camera</small></div>"
                        ), "", "", session_records
                    emotion = max(probs, key=probs.get)
                    col = EMOTION_COLOR[emotion]
                    vi  = EMOTION_VI[emotion]
                    conf = probs[emotion]
                    if bbox:
                        frame = draw_face_label(frame, bbox, f'{vi} {conf:.0%}')
                    out_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label_html = (
                        f'<div style="text-align:center;padding:16px;background:#1a1a2e;'
                        f'border-radius:12px;border:2px solid {col};margin-top:8px">'
                        f'<div style="font-size:40px">{vi.split()[0]}</div>'
                        f'<div style="font-size:20px;font-weight:700;color:{col}">{" ".join(vi.split()[1:])}</div>'
                        f'<div style="color:#aaa;font-size:14px;margin-top:4px">Độ tin cậy: {conf:.0%}</div></div>'
                    )
                    if user_name.strip():
                        record = {
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'date': str(date.today()),
                            'text': '',
                            'text_emotion': None,
                            'text_conf': 0.0,
                            'face_emotion': emotion,
                            'face_conf': conf,
                            'final_emotion': emotion,
                            'final_conf': conf,
                            'w_text': 0.0, 'w_face': 1.0,
                            'conflict': False,
                        }
                        session_records = session_records + [record]
                        history = load_history()
                        history.setdefault(user_name, {}).setdefault(str(date.today()), []).append(record)
                        try:
                            save_history(history)
                            print(f'[SAVED webcam] user={user_name}')
                        except Exception as se:
                            print(f'[SAVE ERROR] {se}')
                    return out_img, label_html, make_bar_html(probs), make_suggestion_html(emotion), session_records
                except Exception as e:
                    print(f'[ERROR webcam] {e}')
                    return None, f"<div style='color:#E84040;padding:10px'>❌ Lỗi: {e}</div>", "", "", session_records

            webcam_btn.click(
                fn=handle_webcam_tab,
                inputs=[webcam_input, user_name_input, session_state],
                outputs=[webcam_out, webcam_label, webcam_bars, webcam_suggestion, session_state]
            )

        # ══════════════════════════════════════════════════════════
        # ── TAB 3: GIỌNG NÓI + KHUÔN MẶT — 1 NÚT DUY NHẤT ───────
        #
        #  GIẢI PHÁP ĐÚNG: Custom JS dùng getUserMedia({video:true, audio:true})
        #  → 1 stream duy nhất cho cả camera + mic → không bao giờ conflict.
        #  Người dùng chỉ bấm 1 nút:
        #    - Lần 1: bắt đầu quay + ghi âm
        #    - Lần 2: dừng → JS tự chụp frame + encode audio → gửi vào 2 hidden
        #      Textbox (base64 image + base64 wav) → Python xử lý bình thường.
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🔀 Giọng nói + Khuôn mặt"):
            gr.HTML('''
<div style="background:#1a1a2e;border-radius:10px;padding:14px 16px;margin:8px 0;border:1px solid #2a2a4e">
  <div style="color:#a78bfa;font-weight:700;font-size:14px;margin-bottom:8px">🎬 Ghi âm + Quét mặt — 1 nút</div>
  <div style="color:#8888aa;font-size:13px;line-height:1.7">
    ① Chọn số giây muốn ghi &nbsp;→&nbsp;
    ② Nhấn nút &nbsp;→&nbsp;
    ③ <b style="color:#e0e0e0">Nhìn thẳng vào camera và nói</b> trong thời gian đó &nbsp;→&nbsp;
    ④ Kết quả tự hiện<br>
    <span style="color:#555;font-size:12px">⚙️ Python mở camera + mic trực tiếp — không qua trình duyệt</span>
  </div>
</div>''')

            with gr.Row():
                with gr.Column(scale=1):
                    mm_seconds = gr.Slider(minimum=3, maximum=15, value=7, step=1,
                                           label="⏱ Thời gian ghi âm (giây) — nói rõ ràng trong khoảng này")
                    mm_btn = gr.Button("🎬  Bắt đầu ghi âm + quét mặt", variant="primary", size="lg")
                    mm_status_box = gr.Textbox(label="Trạng thái", interactive=False, lines=2)

            with gr.Row():
                with gr.Column(scale=2):
                    mm_face_img = gr.Image(label="📸 Ảnh khuôn mặt", interactive=False)
                with gr.Column(scale=3):
                    gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:13px;margin-bottom:4px">🧠 Kết quả tổng hợp</div>')
                    mm_emotion_label = gr.HTML()
                    mm_bars          = gr.HTML()

            gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:14px;margin:14px 0 6px">📝 Văn bản nhận dạng</div>')
            mm_transcribed = gr.Textbox(label="", interactive=False, lines=2)
            gr.HTML('<div style="color:#a78bfa;font-weight:600;font-size:14px;margin:10px 0 6px">💡 Gợi ý cho bạn</div>')
            mm_suggestion = gr.HTML()
            mm_score      = gr.HTML()

            def handle_mm_capture(seconds, user_name, session_records, history):
                """
                Dùng OpenCV + PyAudio để capture trực tiếp từ thiết bị.
                - Yield countdown vào mm_status_box mỗi giây (các component khác giữ nguyên).
                - Camera chụp ảnh ở giữa thời gian ghi âm.
                """
                import time, wave, io

                # Hằng số placeholder để giữ nguyên các output khác khi yield countdown
                _KEEP = gr.update()

                try:
                    import pyaudio
                except ImportError:
                    yield "⚠️ Cần cài: pip install pyaudio", _KEEP, _KEEP, _KEEP, _KEEP, _KEEP, _KEEP, session_records, history
                    return

                seconds = int(seconds)
                if not user_name.strip():
                    user_name = "Người dùng"

                RATE, CHUNK, CHANNELS = 16000, 1024, 1
                snap_at_chunk = int(RATE / CHUNK * seconds / 2)  # chụp ảnh ở giữa

                # ── Mở camera trước (warm-up), giữ mở suốt quá trình ghi ──
                face_frame_bgr = None
                cap = None
                try:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        for _ in range(5):
                            cap.read()
                except Exception as ce:
                    print(f"[WARN cam init] {ce}")
                    cap = None

                # ── Ghi âm + chụp ảnh giữa chừng + countdown ──────────────
                audio_np = None
                pa = None
                try:
                    pa = pyaudio.PyAudio()
                    stream = pa.open(
                        format=pyaudio.paInt16, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK
                    )
                    frames = []
                    n_chunks = int(RATE / CHUNK * seconds)
                    last_reported_sec = -1

                    for i in range(n_chunks):
                        frames.append(stream.read(CHUNK, exception_on_overflow=False))

                        # Cập nhật countdown mỗi giây — chỉ status box thay đổi
                        elapsed_sec = int(i * CHUNK / RATE)
                        remaining = seconds - elapsed_sec
                        if elapsed_sec != last_reported_sec:
                            last_reported_sec = elapsed_sec
                            yield (f"🎙️ Đang ghi âm... còn {remaining} giây",
                                   _KEEP, _KEEP, _KEEP, _KEEP, _KEEP, _KEEP,
                                   session_records, history)

                        # Chụp ảnh đúng giữa thời gian ghi
                        if i == snap_at_chunk and cap is not None and cap.isOpened():
                            try:
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    face_frame_bgr = frame.copy()
                            except Exception as ce:
                                print(f"[WARN cam snap] {ce}")

                    stream.stop_stream(); stream.close()
                    raw = b"".join(frames)
                    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    yield ("⏳ Đang phân tích...",
                           _KEEP, _KEEP, _KEEP, _KEEP, _KEEP, _KEEP,
                           session_records, history)
                except Exception as ae:
                    yield (f"⚠️ Lỗi mic: {ae}\nCài PyAudio: pip install pyaudio",
                           _KEEP, _KEEP, _KEEP, _KEEP, _KEEP, _KEEP,
                           session_records, history)
                    return
                finally:
                    if pa:
                        try: pa.terminate()
                        except: pass
                    if cap:
                        try: cap.release()
                        except: pass

                # ── Nhận dạng giọng nói ───────────────────────────
                text = transcribe_from_gradio_audio((RATE, (audio_np * 32768).astype(np.int16)))
                if not text or text.startswith("⚠️") or text.startswith("Lỗi"):
                    yield text, None, "", "", "", "", session_records, history
                    return

                text_probs = predict_text_full(text)

                # ── Nhận diện khuôn mặt ──────────────────────────
                face_probs = None
                face_display = None
                if face_frame_bgr is not None and face_available:
                    try:
                        face_probs, bbox = predict_face_single(face_frame_bgr)
                        if bbox and face_probs:
                            emotion = max(face_probs, key=face_probs.get)
                            labeled = draw_face_label(face_frame_bgr, bbox, EMOTION_VI[emotion])
                            face_display = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
                        else:
                            face_display = cv2.cvtColor(face_frame_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as fe:
                        print(f"[WARN face] {fe}")
                        face_display = cv2.cvtColor(face_frame_bgr, cv2.COLOR_BGR2RGB)
                        face_probs = None

                # ── Tổng hợp ─────────────────────────────────────
                final_emotion, final_conf, fused, w_text, w_face = fuse_emotions(text_probs, face_probs)
                text_top = max(text_probs, key=text_probs.get)
                face_top = max(face_probs, key=face_probs.get) if face_probs else None
                conflict = bool(face_probs and text_top in POSITIVE_EMOTIONS and face_top in NEGATIVE_EMOTIONS)

                record = {
                    'time': datetime.now().strftime('%H:%M:%S'), 'date': str(date.today()),
                    'text': text, 'text_emotion': text_top, 'text_conf': text_probs[text_top],
                    'face_emotion': face_top,
                    'face_conf': face_probs[face_top] if face_probs and face_top else 0.0,
                    'final_emotion': final_emotion, 'final_conf': final_conf,
                    'w_text': w_text, 'w_face': w_face, 'conflict': conflict,
                }
                session_records = session_records + [record]
                history = load_history()
                history.setdefault(user_name, {}).setdefault(str(date.today()), []).append(record)
                try: save_history(history)
                except Exception as e: print(f"[SAVE ERROR] {e}")

                col = EMOTION_COLOR[final_emotion]; vi = EMOTION_VI[final_emotion]
                detail = (
                    f'<div style="background:#1a1a2e;border-radius:10px;padding:12px;font-size:13px;margin-top:8px">'
                    f'<div style="color:#aaa">📝 Giọng: <span style="color:{EMOTION_COLOR[text_top]}">{EMOTION_VI[text_top]} ({text_probs[text_top]:.0%})</span></div>'
                    + (
                        f'<div style="color:#aaa;margin-top:4px">📷 Mặt: <span style="color:{EMOTION_COLOR.get(face_top,"#aaa")}">{EMOTION_VI.get(face_top,"—")} ({record["face_conf"]:.0%})</span></div>'
                        f'<div style="color:#aaa;margin-top:4px">⚖️ Trọng số: Speech={w_text:.2f} / Mặt={w_face:.2f}</div>'
                        if face_top else
                        '<div style="color:#555;margin-top:4px">📷 Không phát hiện khuôn mặt</div>'
                    )
                    + ('<div style="color:#FF6B6B;margin-top:6px">⚠️ Hidden Mood Detected!</div>' if conflict else '')
                    + '</div>'
                )
                label = (
                    f'<div style="text-align:center;padding:20px;background:#1a1a2e;border-radius:12px;border:2px solid {col}">'
                    f'<div style="font-size:42px">{vi.split()[0]}</div>'
                    f'<div style="font-size:20px;font-weight:700;color:{col}">{" ".join(vi.split()[1:])}</div>'
                    f'<div style="color:#aaa;font-size:14px">Độ tin cậy: {final_conf:.0%}</div>'
                    f'</div>{detail}'
                )
                probs_out = fused if face_probs else text_probs
                bars      = make_bar_html(probs_out)
                suggestion = make_suggestion_html(final_emotion)
                score      = make_score_html(user_name, history) + check_alerts(session_records)
                status_msg = f"✅ Xong! Cảm xúc: {vi} ({final_conf:.0%})"
                yield status_msg, face_display, text, label, bars, suggestion, score, session_records, history

            mm_btn.click(
                fn=handle_mm_capture,
                inputs=[mm_seconds, user_name_input, session_state, history_state],
                outputs=[mm_status_box, mm_face_img, mm_transcribed,
                         mm_emotion_label, mm_bars, mm_suggestion, mm_score,
                         session_state, history_state]
            )

        # ── TAB 4: LỊCH SỬ ───────────────────────────────────────
        with gr.Tab("📊 Lịch sử"):
            gr.HTML('<div style="color:#a78bfa;font-size:13px;padding:8px 0 4px">Nhấn nút để xem biểu đồ cảm xúc của bạn</div>')
            history_btn = gr.Button("🔄 Xem biểu đồ cảm xúc", variant="primary")
            history_out = gr.HTML()

            history_btn.click(
                fn=lambda name: handle_history(name),
                inputs=[user_name_input],
                outputs=[history_out]
            )

# ══════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  MENTAL HEALTH MONITOR")
    print("="*60)
    print(f"  Text model  : ✅")
    print(f"  Face model  : {'✅' if face_available else '⚠️  không tìm thấy'}")
    print(f"  Speech model: {'✅ PhoWhisper' if speech_available else '⚠️  không tìm thấy'}")
    print("="*60)
    print(f"  Gradio UI   : http://localhost:7860")
    print("="*60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )

# python mental_health_ui_v2.py