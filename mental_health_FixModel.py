"""
Mental Health Monitor — Gradio UI
Tích hợp: Text Emotion + Face Emotion + Speech-to-Text (PhoWhisper)
Chạy: python mental_health_ui.py
"""

import torch
import torch.nn as nn
import cv2
import json
import os
import re
import random
import time
import threading
import numpy as np
from datetime import datetime, date
from collections import Counter, deque
from torchvision import models, transforms
from transformers import (
    XLMRobertaForSequenceClassification, AutoTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration, pipeline
)
import gradio as gr

# ══════════════════════════════════════════════════════════════
#  CẤU HÌNH — chỉnh đường dẫn tại đây
# ══════════════════════════════════════════════════════════════
TEXT_MODEL_PATH  = r"emotion_model_v5.3"
FACE_MODEL_PATH  = r"nb_resnet_best.pth"
SPEECH_MODEL_PATH = r"phowhisper_small_vietsuperspeech"  # ← folder model vừa train
HISTORY_FILE     = r"mental_health_history.json"

TEMPORAL_FRAMES   = 15
ALERT_CONSECUTIVE = 3
ALERT_DAY_PCT     = 0.6
TREND_DAYS        = 3
ALERT_CONF        = 0.65
MIN_FRAMES_DECIDE = 5

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
COMMON_EMOTIONS = ['sadness','joy','love','anger','fear','surprise','neutral']
TEXT_EMOTIONS   = ['sadness','joy','love','anger','fear','surprise','neutral']
FACE_EMOTIONS   = ['surprise','fear','disgust','happiness','sadness','anger','neutral']
FACE_TO_COMMON  = {'happiness':'joy','sadness':'sadness','anger':'anger','fear':'fear',
                   'surprise':'surprise','neutral':'neutral','disgust':'anger'}
EMOTION_VI      = {'sadness':'😢 Buồn bã','joy':'😊 Vui vẻ','love':'🥰 Yêu thương',
                   'anger':'😠 Tức giận','fear':'😰 Lo lắng','surprise':'😲 Ngạc nhiên',
                   'neutral':'😐 Bình thường'}
EMOTION_COLOR   = {'sadness':'#6B9BD2','joy':'#F5A623','love':'#E85D9A',
                   'anger':'#E84040','fear':'#9B59B6','surprise':'#27AE60','neutral':'#7F8C8D'}
NEGATIVE_EMOTIONS = {'sadness','anger','fear'}
POSITIVE_EMOTIONS = {'joy','love','surprise'}

MUSIC = {
    'sadness':[
        {'title':'Tâm Sự Tuổi 30 - Trịnh Thăng Bình','url':'https://youtu.be/kV3famkRaA4'},
        {'title':'Chỉ Là Không Cùng Nhau - Tăng Phúc','url':'https://youtu.be/xBjJFoDK1Zw'},
        {'title':'Someone Like You - Adele','url':'https://youtu.be/hLQl3WQQoQ0'},
        {'title':'Fix You - Coldplay','url':'https://youtu.be/k4V3Mo61fJM'},
        {'title':'Sau Tất Cả - Erik','url':'https://youtu.be/wHF3Jv6Gk2o'},
    ],
    'joy':[
        {'title':'Hãy Trao Cho Anh - Sơn Tùng M-TP','url':'https://youtu.be/knW7-x7Y7RE'},
        {'title':'Happy - Pharrell Williams','url':'https://youtu.be/ZbZSe6N_BXs'},
        {'title':'Good as Hell - Lizzo','url':'https://youtu.be/SmbmeOgWsqE'},
        {'title':'Đây là một bài hát vui - Jun Phạm','url':'https://youtu.be/lZ8Ru-hAg9s'},
        {'title':'Cứ vui lên - Mỹ Tâm','url':'https://youtu.be/y70kmGVY2tA'},
    ],
    'love':[
        {'title':'Yêu Được Không - Đức Phúc','url':'https://youtu.be/_VGm6brq1aI'},
        {'title':'Perfect - Ed Sheeran','url':'https://youtu.be/cNGjD0VG4R8'},
        {'title':'Yêu 5 - Rymastic','url':'https://youtu.be/QFQdIvKSQ2Q'},
        {'title':'Tháng 4 Là Lời Nói Dối - Hà Anh Tuấn','url':'https://youtu.be/UCXao7aTDQM'},
        {'title':'A Thousand Years - C.Perri','url':'https://youtu.be/rtOvBOTyX00'},
    ],
    'anger':[
        {'title':'Hít vào thở ra - Min x Hieuthuhai','url':'https://youtu.be/Q3xlEH3_HGA'},
        {'title':'Những ngày trời bao la - Bùi Công Nam','url':'https://youtu.be/q0tvU2MFyVA'},
        {'title':'Một Ngày Chẳng Nắng - Pháo','url':'https://youtu.be/ABuY4KUUVcI'},
        {'title':'Mùa hè tuyệt vời - Đức Phúc x Tăng Duy Tân','url':'https://youtu.be/2YoIKPOUwIM'},
        {'title':'Cứ Chill Thôi - Chillies','url':'https://youtu.be/LZN4I3K8SC0'},
    ],
    'fear':[
        {'title':'Xe đạp - Thuỳ Chi','url':'https://youtu.be/6KJrNWC0tfw'},
        {'title':'Breathe (2AM) - Anna Nalick','url':'https://youtu.be/FcvXr-9XtgA'},
        {'title':'Mọi chuyện rồi cũng sẽ qua - duongw','url':'https://youtu.be/7ssyAFpQqCg'},
        {'title':'Shape of You - Ed Sheeran','url':'https://youtu.be/JGwWNGJdvx8'},
        {'title':'Sẽ Ổn Thôi - Khải','url':'https://youtu.be/TyZasCMDf5M'},
    ],
    'surprise':[
        {'title':'Shake It Off - Taylor Swift','url':'https://youtu.be/nfWlot6h_JM'},
        {'title':'Sáng mắt chưa? - Trúc Nhân','url':'https://youtu.be/rDhx4ejrPPA'},
        {'title':'Thật bất ngờ - Trúc Nhân','url':'https://youtu.be/YUAmi7Q2F0Y'},
        {'title':'Vũ điệu cồng chiêng - Tóc Tiên','url':'https://youtu.be/Rz4FbACtfd0'},
        {'title':'GANGNAM STYLE - PSY','url':'https://youtu.be/9bZkp7q19f0'},
    ],
    'neutral':[
        {'title':'Không Cảm Xúc - Hồ Quang Hiếu','url':'https://youtu.be/YZIjQDZl6Ko'},
        {'title':'Weightless - Marconi Union','url':'https://youtu.be/UfcAVejslrU'},
        {'title':'Vì tôi còn sống - Tiên Tiên','url':'https://youtu.be/Of-UkRiRWeo'},
        {'title':'Việt Nam những chuyến đi - Vicky Nhung','url':'https://youtu.be/46EjkkDo00g'},
        {'title':'Trốn Tìm - Đen Vâu','url':'https://youtu.be/Ws-QlpSltr8'},
    ],
}

BREATHING = {
    'sadness':{'name':'Thở 4-7-8 (Thư giãn sâu)','steps':['Ngồi thẳng lưng, thả lỏng vai','Hít vào qua mũi 4 giây','Nín thở 7 giây','Thở ra qua miệng 8 giây','Lặp lại 4 lần, 2 lần/ngày']},
    'anger':{'name':'Thở hộp (Box Breathing)','steps':['Ngồi thoải mái, thở ra hết','Hít vào 4 giây','Nín thở 4 giây','Thở ra 4 giây','Nín thở 4 giây','Lặp lại 4-6 lần']},
    'fear':{'name':'Thở 5-5-5 (Chống lo âu)','steps':['Nhắm mắt, tập trung hơi thở','Hít vào 5 giây','Giữ 5 giây','Thở ra 5 giây','Nhủ thầm "Tôi an toàn"','Lặp lại 6-10 lần']},
    'joy':{'name':'Thiền biết ơn','steps':['Nhắm mắt, hít thở tự nhiên','Nghĩ 3 điều tốt hôm nay','Cảm nhận sự biết ơn','Hít vào khi nghĩ điều tốt','Thở ra với nụ cười nhẹ','Giữ 5 phút']},
    'love':{'name':'Thiền từ bi','steps':['Ngồi thoải mái, nhắm mắt','Đặt tay lên ngực','Hít vào: "Tôi xứng đáng yêu thương"','Thở ra: gửi yêu thương đến người thân','Lặp lại với từng người','Kết thúc: gửi đến tất cả']},
    'surprise':{'name':'Thở cân bằng','steps':['Ngồi yên, hít thở bình thường','Hít vào đều 4 giây','Thở ra đều 4 giây','Tập trung không khí vào/ra','Dần tăng lên 6-8 giây','Thực hiện 5-10 phút']},
    'neutral':{'name':'Thiền chánh niệm','steps':['Ngồi/nằm thoải mái','Nhắm mắt, tập trung hơi thở','Quan sát bụng phồng/xẹp','Nếu tâm lang thang, đưa về hơi thở','Không phán xét, chỉ quan sát','10-15 phút/ngày']},
}

ACTIVITIES = {
    'sadness':['🚶 Đi bộ 20 phút ngoài trời','📔 Viết nhật ký cảm xúc','☎️ Gọi người thân tin tưởng','🎨 Vẽ/tô màu tự do','🛁 Tắm nước ấm thư giãn'],
    'joy':['💃 Nhảy theo nhạc yêu thích','📸 Chụp ảnh kỷ niệm','🤝 Chia sẻ niềm vui với người thân','🎯 Làm việc quan trọng nhờ năng lượng tích cực','🍳 Nấu món yêu thích'],
    'love':['💌 Nhắn tin cho người bạn yêu','📷 Xem lại ảnh kỷ niệm','🎁 Chuẩn bị bất ngờ nhỏ','🌸 Mua hoa thể hiện tình cảm','📞 Video call nếu xa nhau'],
    'anger':['🥊 Chạy bộ hoặc tập gym','✍️ Viết ra rồi xé tờ giấy đó đi','🧊 Rửa mặt nước lạnh','🚪 Đi bộ nhanh 10 phút','⏰ Đếm ngược từ 10 trước khi phản ứng'],
    'fear':['📝 Viết ra nỗi lo và phân tích thực tế','👥 Tâm sự với người tin tưởng','🎯 Chia nhỏ vấn đề thành bước nhỏ','📚 Đọc sách/xem phim hài','🌙 Chuẩn bị kỹ cho việc sắp xảy ra'],
    'surprise':['📖 Xử lý thông tin từ từ, đừng vội','🫁 Hít sâu 3 lần trước khi quyết định','📞 Hỏi ý kiến người thân','✏️ Viết ưu/nhược điểm tình huống mới','🍵 Pha trà ngồi suy nghĩ bình tĩnh'],
    'neutral':['🎯 Đặt mục tiêu nhỏ cho hôm nay','📚 Đọc sách hoặc nghe podcast','🌿 Dọn dẹp không gian sống','🏃 Vận động nhẹ 15-20 phút','📱 Hỏi thăm người lâu chưa liên lạc'],
}

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
    cfg.update({'model_type':'xlm-roberta','vocab_size':250002,'type_vocab_size':1,'max_position_embeddings':514})
    with open(config_path,'w',encoding='utf-8') as f:
        json.dump(cfg,f,indent=2)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = XLMRobertaForSequenceClassification.from_pretrained(
    TEXT_MODEL_PATH, ignore_mismatched_sizes=True).to(device).eval()
print('✅ Text model loaded!')

# Face model
class Res18Feature(nn.Module):
    def __init__(self, num_classes=7):
        super(Res18Feature, self).__init__()
        self.resnet = models.resnet18(pretrained=False) 
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

print('📥 Loading Face Model...')
face_model = Res18Feature(num_classes=7).to(device)
face_available = False
if os.path.exists(FACE_MODEL_PATH):
    ckpt = torch.load(FACE_MODEL_PATH, map_location=device, weights_only=False)
    sd = {k.replace('module.',''):v for k,v in ckpt['model_state_dict'].items()}
    face_model.load_state_dict(sd, strict=False)
    face_model.eval()
    face_available = True
    print('✅ Face model loaded!')
else:
    print(f'⚠️  Không tìm thấy face model: {FACE_MODEL_PATH}')

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Speech model (PhoWhisper fine-tuned)
print('📥 Loading Speech Model (PhoWhisper)...')
speech_available = False
asr_pipe = None
if os.path.exists(SPEECH_MODEL_PATH):
    try:
        asr_pipe = pipeline(
            'automatic-speech-recognition',
            model=SPEECH_MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            #          generate_kwargs={
            #     "language": "vi",
            #     "task": "transcribe",
            #     "no_repeat_ngram_size": 3,
            #     "repetition_penalty": 1.3,
            #     "num_beams": 5,
            # }
        )
        speech_available = True
        print('✅ Speech model loaded!')
    except Exception as e:
        print(f'⚠️  Lỗi load speech model: {e}')
else:
    print(f'⚠️  Không tìm thấy speech model: {SPEECH_MODEL_PATH}')

# ══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════
def predict_text_full(text):
    inputs = text_tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        probs = torch.softmax(text_model(**inputs).logits, dim=-1)[0].cpu().tolist()
    return {e:p for e,p in zip(TEXT_EMOTIONS, probs)}

def predict_face_single(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    inp = face_transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = face_model(inp)
        probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
    common = {e:0.0 for e in COMMON_EMOTIONS}
    for fl,p in zip(FACE_EMOTIONS, probs):
        common[FACE_TO_COMMON[fl]] += p
    return common, (x,y,w,h)

def resample_audio(arr, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return arr
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(arr, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        n_samples = int(len(arr) / orig_sr * target_sr)
        indices   = np.linspace(0, len(arr) - 1, n_samples)
        return np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)

def transcribe_from_array(arr: np.ndarray, sr: int = 16000) -> str:
    """Nhận dạng từ numpy array — y hệt test_speech.py."""
    if not speech_available or asr_pipe is None:
        return "⚠️ Speech model chưa load"
    try:
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if np.abs(arr).max() > 1.0:
            arr = arr / 32768.0
        if sr != 16000:
            arr = resample_audio(arr, orig_sr=sr, target_sr=16000)
        result = asr_pipe({'array': arr, 
                           'sampling_rate': 16000},
                            generate_kwargs={
                                "language": "vi",
                                "task": "transcribe",
                                "no_repeat_ngram_size": 3,
                                "repetition_penalty": 1.3,
                                "num_beams": 5,
                            })
        text   = result['text']
        text   = re.sub(r'<\|[^|]+\|>', '', text).strip()
        return text
    except Exception as e:
        return f"Lỗi: {e}"

def record_mic_and_transcribe(seconds: int, user_name: str):
    """Ghi âm bằng sounddevice (giống test_speech.py) → phân tích cảm xúc."""
    try:
        import sounddevice as sd
    except ImportError:
        return "❌ Cần cài: pip install sounddevice", "", "", "", ""
    try:
        audio = sd.rec(int(seconds * 16000), samplerate=16000,
                       channels=1, dtype='float32')
        sd.wait()
        arr  = audio.flatten()
        text = transcribe_from_array(arr, 16000)
        if not text or text.startswith("⚠️") or text.startswith("Lỗi"):
            return text, "", "", "", ""
        return handle_text_input(text, user_name)
    except Exception as e:
        return f"Lỗi ghi âm: {e}", "", "", "", ""

def transcribe_audio(audio):
    """Upload file WAV/MP3 qua Gradio."""
    if not speech_available or asr_pipe is None:
        return "⚠️ Speech model chưa load"
    if audio is None:
        return ""
    try:
        sr, arr = audio
        return transcribe_from_array(arr, sr)
    except Exception as e:
        return f"Lỗi: {e}"

def fuse_emotions(text_probs, face_probs=None):
    text_conf = max(text_probs.values())
    face_conf = max(face_probs.values()) if face_probs else 0.0
    if face_probs and (text_conf + face_conf) > 0:
        w_text = text_conf / (text_conf + face_conf)
        w_face = face_conf / (text_conf + face_conf)
    else:
        w_text, w_face = 1.0, 0.0
    fused = {e: w_text*text_probs.get(e,0.0) + w_face*(face_probs.get(e,0.0) if face_probs else 0.0)
             for e in COMMON_EMOTIONS}
    best = max(fused, key=fused.get)
    return best, fused[best], fused, w_text, w_face

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,'r',encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE,'w',encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_health_score(records):
    if not records: return 50
    neg = sum(1 for r in records if r['final_emotion'] in NEGATIVE_EMOTIONS) / len(records)
    pos = sum(1 for r in records if r['final_emotion'] in POSITIVE_EMOTIONS) / len(records)
    return max(0, min(100, int((1-neg)*70 + pos*30)))

def make_bar_html(probs):
    """Tạo HTML bar chart cho emotion probabilities."""
    bars = ""
    for e in COMMON_EMOTIONS:
        p   = probs.get(e, 0.0)
        pct = int(p * 100)
        col = EMOTION_COLOR[e]
        vi  = EMOTION_VI[e]
        bars += f"""
        <div style="display:flex;align-items:center;margin:4px 0;gap:8px">
          <span style="width:120px;font-size:13px;color:#ddd">{vi}</span>
          <div style="flex:1;background:#2a2a3e;border-radius:4px;height:18px;overflow:hidden">
            <div style="width:{pct}%;background:{col};height:100%;border-radius:4px;
                        transition:width 0.4s ease"></div>
          </div>
          <span style="width:38px;text-align:right;font-size:13px;color:{col};font-weight:600">{pct}%</span>
        </div>"""
    return f'<div style="padding:8px 0">{bars}</div>'

def make_suggestion_html(emotion):
    songs = random.sample(MUSIC[emotion], 2)
    b     = BREATHING[emotion]
    acts  = random.sample(ACTIVITIES[emotion], 3)
    col   = EMOTION_COLOR[emotion]

    song_html = "".join(
        f'<a href="{s["url"]}" target="_blank" style="display:block;color:{col};'
        f'text-decoration:none;margin:4px 0;font-size:13px">🎵 {s["title"]}</a>'
        for s in songs
    )
    steps_html = "".join(f'<div style="font-size:13px;color:#ccc;margin:3px 0">• {st}</div>'
                         for st in b['steps'])
    acts_html  = "".join(f'<div style="font-size:13px;color:#ccc;margin:3px 0">{a}</div>'
                         for a in acts)

    return f"""
    <div style="background:#1a1a2e;border-radius:12px;padding:16px;border-left:4px solid {col}">
      <div style="margin-bottom:14px">
        <div style="color:{col};font-weight:700;margin-bottom:6px;font-size:14px">🎵 NHẠC PHÙ HỢP</div>
        {song_html}
      </div>
      <div style="margin-bottom:14px">
        <div style="color:{col};font-weight:700;margin-bottom:6px;font-size:14px">🫁 {b['name']}</div>
        {steps_html}
      </div>
      <div>
        <div style="color:{col};font-weight:700;margin-bottom:6px;font-size:14px">🎯 HOẠT ĐỘNG GỢI Ý</div>
        {acts_html}
      </div>
    </div>"""

# ══════════════════════════════════════════════════════════════
#  SESSION STATE (in-memory)
# ══════════════════════════════════════════════════════════════
session_records = []
history = load_history()

def log_record(user_name, text, text_probs, face_probs=None):
    final_emotion, final_conf, fused, w_text, w_face = fuse_emotions(text_probs, face_probs)
    text_top = max(text_probs, key=text_probs.get)
    face_top = max(face_probs, key=face_probs.get) if face_probs else None
    # Hidden mood check
    conflict = False
    if face_probs:
        t = max(text_probs, key=text_probs.get)
        f = max(face_probs, key=face_probs.get)
        conflict = t in POSITIVE_EMOTIONS and f in NEGATIVE_EMOTIONS

    record = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'date': str(date.today()),
        'text': text,
        'text_emotion': text_top,
        'text_conf': text_probs[text_top],
        'face_emotion': face_top,
        'face_conf': face_probs[face_top] if face_probs and face_top else 0.0,
        'final_emotion': final_emotion,
        'final_conf': final_conf,
        'w_text': w_text, 'w_face': w_face,
        'conflict': conflict,
    }
    session_records.append(record)
    today = str(date.today())
    history.setdefault(user_name, {}).setdefault(today, []).append(record)
    save_history(history)
    return record

# ══════════════════════════════════════════════════════════════
#  GRADIO HANDLERS
# ══════════════════════════════════════════════════════════════

def handle_speech_input(audio, user_name):
    """Xử lý input giọng nói: transcribe → text → analyze."""
    text = transcribe_audio(audio)
    if not text or text.startswith("⚠️") or text.startswith("Lỗi"):
        return text, "", "", "", ""
    # Tiếp tục phân tích text vừa transcribe
    return handle_text_input(text, user_name)

def handle_text_input(text, user_name):
    """Phân tích text và trả kết quả."""
    if not text.strip():
        return "", "", "", "", ""
    if not user_name.strip():
        user_name = "Người dùng"

    text_probs = predict_text_full(text)
    record = log_record(user_name, text, text_probs, None)
    emotion = record['final_emotion']

    # Emotion label
    col   = EMOTION_COLOR[emotion]
    vi    = EMOTION_VI[emotion]
    conf  = record['final_conf']
    label = f'<div style="text-align:center;padding:20px;background:#1a1a2e;border-radius:12px;border:2px solid {col}">' \
            f'<div style="font-size:42px;margin-bottom:8px">{vi.split()[0]}</div>' \
            f'<div style="font-size:20px;font-weight:700;color:{col}">{" ".join(vi.split()[1:])}</div>' \
            f'<div style="font-size:14px;color:#aaa;margin-top:4px">Độ tin cậy: {conf:.0%}</div>' \
            f'{"<div style=color:#FF6B6B;margin-top:6px>⚠️ Hidden Mood Detected!</div>" if record["conflict"] else ""}' \
            f'</div>'

    bars       = make_bar_html(text_probs)
    suggestion = make_suggestion_html(emotion)
    score_html = make_score_html(user_name)
    alert_html = check_alerts(user_name)

    return text, label, bars, suggestion, score_html + alert_html

def handle_webcam_input(image, user_name):
    """Phân tích ảnh webcam — nhận numpy array (RGB) từ Gradio."""
    if image is None:
        return None, "<div style='color:#F5A623;padding:10px;background:#1a1a2e;border-radius:8px'>⚠️ Chưa có ảnh — nhấn 📷 để chụp snapshot hoặc upload ảnh</div>", "", ""
    if not face_available:
        return None, "<div style='color:#E84040;padding:10px'>⚠️ Face model chưa load</div>", "", ""

    try:
        # Gradio type="numpy" → RGB uint8 array
        img_arr = np.array(image)
        if img_arr.ndim == 2:                      # grayscale
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:                # RGBA
            img_arr = img_arr[:, :, :3]

        frame = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        probs, bbox = predict_face_single(frame)

        if probs is None:
            return img_arr, (
                "<div style='color:#aaa;padding:12px;background:#1a1a2e;border-radius:10px'>"
                "😶 Không phát hiện khuôn mặt<br>"
                "<small style='color:#666'>Thử chụp gần hơn, đủ ánh sáng, nhìn thẳng vào camera</small></div>"
            ), "", ""

        emotion = max(probs, key=probs.get)
        col     = EMOTION_COLOR[emotion]
        vi      = EMOTION_VI[emotion]
        conf    = probs[emotion]

        # Vẽ bounding box lên ảnh
        if bbox:
            x, y, w, h = bbox
            color_bgr   = (0, 0, 220) if emotion in NEGATIVE_EMOTIONS else (0, 200, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
            cv2.putText(frame, f'{vi} {conf:.0%}',
                        (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_bgr, 2)

        out_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        label_html = (
            f'<div style="text-align:center;padding:16px;background:#1a1a2e;'
            f'border-radius:12px;border:2px solid {col};margin-top:8px">'
            f'<div style="font-size:40px">{vi.split()[0]}</div>'
            f'<div style="font-size:20px;font-weight:700;color:{col}">{" ".join(vi.split()[1:])}</div>'
            f'<div style="color:#aaa;font-size:14px;margin-top:4px">Độ tin cậy: {conf:.0%}</div>'
            f'</div>'
        )
        bars       = make_bar_html(probs)
        suggestion = make_suggestion_html(emotion)
        return out_img, label_html, bars, suggestion

    except Exception as e:
        return None, f"<div style='color:#E84040;padding:10px;background:#1a1a2e;border-radius:8px'>❌ Lỗi: {e}</div>", "", ""

def handle_multimodal(text, audio, image, user_name):
    """Kết hợp text + speech + face."""
    if not user_name.strip():
        user_name = "Người dùng"

    # Speech → text nếu có audio
    if audio is not None and speech_available:
        transcribed = transcribe_audio(audio)
        if transcribed and not transcribed.startswith("⚠️") and not transcribed.startswith("Lỗi"):
            text = transcribed if not text.strip() else text + " " + transcribed

    if not text.strip():
        return "", "<div style='color:#aaa;padding:10px'>Nhập text hoặc ghi âm để phân tích</div>", "", "", ""

    text_probs = predict_text_full(text)

    # Face probs nếu có ảnh
    face_probs = None
    if image is not None and face_available:
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                img_arr = np.array(image.convert('RGB'))
            else:
                img_arr = np.array(image)
            frame = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            face_probs, _ = predict_face_single(frame)
        except Exception:
            face_probs = None

    record = log_record(user_name, text, text_probs, face_probs)
    emotion = record['final_emotion']
    col     = EMOTION_COLOR[emotion]
    vi      = EMOTION_VI[emotion]

    # Detail panel
    detail = f"""
    <div style="background:#1a1a2e;border-radius:12px;padding:14px;font-size:13px">
      <div style="color:#aaa;margin-bottom:6px">📝 Text: <span style="color:{EMOTION_COLOR[record['text_emotion']]}">{EMOTION_VI[record['text_emotion']]} ({record['text_conf']:.0%})</span></div>
      {"<div style='color:#aaa;margin-bottom:6px'>📷 Mặt: <span style='color:" + EMOTION_COLOR.get(record['face_emotion'],'#aaa') + "'>" + EMOTION_VI.get(record['face_emotion'],'') + f" ({record['face_conf']:.0%})</span></div>" if record['face_emotion'] else ""}
      {"<div style='color:#aaa;margin-bottom:6px'>⚖️ Trọng số: Text={record['w_text']:.2f} / Mặt={record['w_face']:.2f}</div>" if record['face_emotion'] else ""}
      {"<div style='color:#FF6B6B'>⚠️ Hidden Mood Detected!</div>" if record['conflict'] else ""}
    </div>"""

    label = f'<div style="text-align:center;padding:20px;background:#1a1a2e;border-radius:12px;border:2px solid {col}">' \
            f'<div style="font-size:42px">{vi.split()[0]}</div>' \
            f'<div style="font-size:20px;font-weight:700;color:{col}">{" ".join(vi.split()[1:])}</div>' \
            f'<div style="color:#aaa;font-size:14px">Độ tin cậy: {record["final_conf"]:.0%}</div>' \
            f'</div>'

    bars = make_bar_html(record.get('fused', text_probs) if not face_probs else
                         {e: record['w_text']*text_probs.get(e,0) + record['w_face']*face_probs.get(e,0)
                          for e in COMMON_EMOTIONS})
    suggestion = make_suggestion_html(emotion)
    score_html = make_score_html(user_name)

    return text, label + detail, bars, suggestion, score_html

def make_score_html(user_name):
    today = str(date.today())
    records = history.get(user_name, {}).get(today, [])
    if not records:
        return ""
    score  = get_health_score(records)
    status = '🟢 Ổn định' if score >= 65 else '🟡 Cần theo dõi' if score >= 40 else '🔴 Cần hỗ trợ'
    col    = '#27AE60' if score >= 65 else '#F5A623' if score >= 40 else '#E84040'
    counts = Counter(r['final_emotion'] for r in records)
    dominant = counts.most_common(1)[0][0]
    neg_pct  = sum(counts.get(e,0) for e in NEGATIVE_EMOTIONS) / len(records) * 100

    # Circular score
    dash = int(score * 2.51)  # circumference ≈ 251
    return f"""
    <div style="background:#1a1a2e;border-radius:12px;padding:16px;margin-top:12px">
      <div style="color:#fff;font-weight:700;margin-bottom:10px;font-size:14px">📊 SỨC KHỎE HÔM NAY</div>
      <div style="display:flex;align-items:center;gap:20px">
        <svg width="80" height="80" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r="32" fill="none" stroke="#2a2a3e" stroke-width="8"/>
          <circle cx="40" cy="40" r="32" fill="none" stroke="{col}" stroke-width="8"
                  stroke-dasharray="{dash} 251" stroke-linecap="round"
                  transform="rotate(-90 40 40)"/>
          <text x="40" y="45" text-anchor="middle" fill="{col}" font-size="16" font-weight="bold">{score}</text>
        </svg>
        <div>
          <div style="color:{col};font-weight:700;font-size:15px">{status}</div>
          <div style="color:#aaa;font-size:12px;margin-top:4px">Chủ đạo: {EMOTION_VI[dominant]}</div>
          <div style="color:#aaa;font-size:12px">Tiêu cực: {neg_pct:.0f}% ({len(records)} lượt)</div>
        </div>
      </div>
    </div>"""

def check_alerts(user_name):
    """Kiểm tra cảnh báo liên tục."""
    if len(session_records) < ALERT_CONSECUTIVE:
        return ""
    last = session_records[-ALERT_CONSECUTIVE:]
    if all(r['final_emotion'] in NEGATIVE_EMOTIONS and r['final_conf'] >= ALERT_CONF for r in last):
        em = last[-1]['final_emotion']
        col = EMOTION_COLOR[em]
        return f"""<div style="background:#2a0a0a;border:1px solid #E84040;border-radius:10px;padding:12px;margin-top:10px">
          <div style="color:#E84040;font-weight:700">🚨 CẢNH BÁO</div>
          <div style="color:#ccc;font-size:13px;margin-top:4px">{EMOTION_VI[em]} liên tục {ALERT_CONSECUTIVE} lần. Hãy thử bài tập thở hoặc liên hệ người thân.</div>
        </div>"""
    return ""

def get_history_html(user_name):
    """Lấy lịch sử 14 ngày."""
    if not user_name.strip() or user_name not in history:
        return "<div style='color:#aaa;padding:10px'>Chưa có dữ liệu</div>"
    dates  = sorted(history[user_name].keys())[-14:]
    rows   = ""
    for d in reversed(dates):
        recs   = history[user_name][d]
        score  = get_health_score(recs)
        dom    = Counter(r['final_emotion'] for r in recs).most_common(1)[0][0]
        col    = '#27AE60' if score >= 65 else '#F5A623' if score >= 40 else '#E84040'
        icon   = '🟢' if score >= 65 else '🟡' if score >= 40 else '🔴'
        rows  += f"""<tr>
          <td style="padding:8px 12px;color:#aaa;font-size:13px">{d}</td>
          <td style="padding:8px 12px;color:{col};font-weight:700">{icon} {score}/100</td>
          <td style="padding:8px 12px;color:#ddd;font-size:13px">{EMOTION_VI[dom]}</td>
          <td style="padding:8px 12px;color:#aaa;font-size:13px">{len(recs)} lượt</td>
        </tr>"""
    return f"""<table style="width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:10px;overflow:hidden">
      <tr style="background:#0d0d1a">
        <th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:12px">NGÀY</th>
        <th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:12px">ĐIỂM SK</th>
        <th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:12px">CẢM XÚC CHỦ ĐẠO</th>
        <th style="padding:10px 12px;color:#7c7caa;text-align:left;font-size:12px">LƯỢT</th>
      </tr>
      {rows}
    </table>"""

# ══════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════
CSS = """
body, .gradio-container { background: #0d0d1a !important; color: #e0e0e0 !important; font-family: 'Segoe UI', sans-serif; }
.tab-nav button { background: #1a1a2e !important; color: #8888aa !important; border: none !important; padding: 10px 20px !important; font-size: 14px !important; }
.tab-nav button.selected { color: #a78bfa !important; border-bottom: 2px solid #a78bfa !important; background: #1a1a2e !important; }
textarea, input[type=text] { background: #1a1a2e !important; color: #e0e0e0 !important; border: 1px solid #2a2a4e !important; border-radius: 8px !important; }
.gr-button { background: linear-gradient(135deg,#6366f1,#a855f7) !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.gr-button:hover { opacity: 0.9 !important; transform: translateY(-1px); }
label { color: #8888aa !important; font-size: 13px !important; }
.gr-panel { background: #13132a !important; border: 1px solid #2a2a4e !important; border-radius: 12px !important; }
"""

HEADER_HTML = """
<div style="text-align:center;padding:24px 0 16px;background:linear-gradient(135deg,#0d0d1a,#1a1a2e)">
  <div style="font-size:36px;margin-bottom:4px">🧠</div>
  <div style="font-size:24px;font-weight:700;color:#a78bfa;letter-spacing:1px">MENTAL HEALTH MONITOR</div>
  <div style="font-size:13px;color:#666;margin-top:4px">Text · Giọng nói · Khuôn mặt · Đa phương thức</div>
</div>"""

with gr.Blocks(title="Mental Health Monitor", css=CSS) as app:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        user_name_input = gr.Textbox(label="👤 Tên người dùng", placeholder="Nhập tên...",
                                     scale=2, max_lines=1)

    with gr.Tabs():
        # ── TAB 1: TEXT ──────────────────────────────────────────
        with gr.Tab("💬 Text"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Nhập cảm xúc / tâm trạng",
                                            placeholder="Hôm nay tôi cảm thấy...",
                                            lines=4)
                    text_btn = gr.Button("🔍 Phân tích", variant="primary")

                with gr.Column(scale=3):
                    text_emotion_label = gr.HTML(label="Cảm xúc")
                    text_bars          = gr.HTML(label="Phân bố")

            text_suggestion = gr.HTML(label="Gợi ý")
            text_score_alert = gr.HTML()

            text_btn.click(
                fn=handle_text_input,
                inputs=[text_input, user_name_input],
                outputs=[text_input, text_emotion_label, text_bars, text_suggestion, text_score_alert]
            )

        # ── TAB 2: GIỌNG NÓI ────────────────────────────────────
        with gr.Tab("🎙️ Giọng nói"):
            gr.HTML(f'<div style="color:{"#27AE60" if speech_available else "#E84040"};padding:8px 8px 0;font-size:13px">'
                    f'{"✅ PhoWhisper sẵn sàng (WER ~18.7%)" if speech_available else "⚠️ Speech model chưa load"}</div>')
            gr.HTML('<div style="color:#8888aa;font-size:12px;padding:4px 8px 12px">'
                    '🎙️ Ghi âm trực tiếp bằng sounddevice (chất lượng tốt nhất) hoặc upload file WAV</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    # Ghi âm bằng sounddevice
                    gr.HTML('<div style="color:#a78bfa;font-size:13px;font-weight:600;margin-bottom:6px">🎙️ Ghi âm trực tiếp</div>')
                    with gr.Row():
                        rec_3s  = gr.Button("⏺ 3 giây",  variant="secondary", scale=1)
                        rec_5s  = gr.Button("⏺ 5 giây",  variant="secondary", scale=1)
                        rec_10s = gr.Button("⏺ 10 giây", variant="secondary", scale=1)

                    gr.HTML('<div style="color:#666;font-size:12px;margin:8px 0">── hoặc upload file WAV/MP3 ──</div>')
                    audio_input = gr.Audio(label="📁 Upload file", type="numpy",
                                           sources=["upload"])
                    upload_btn  = gr.Button("🔍 Nhận dạng file", variant="secondary")

                with gr.Column(scale=3):
                    speech_transcribed   = gr.Textbox(
                        label="📝 Văn bản nhận dạng", interactive=False,
                        placeholder="Văn bản sẽ hiện ở đây...")
                    speech_emotion_label = gr.HTML()
                    speech_bars          = gr.HTML()

            speech_suggestion  = gr.HTML()
            speech_score_alert = gr.HTML()

            # Ghi âm sounddevice
            rec_3s.click(fn=lambda u: record_mic_and_transcribe(3, u),
                          inputs=[user_name_input],
                          outputs=[speech_transcribed, speech_emotion_label, speech_bars,
                                   speech_suggestion, speech_score_alert])
            rec_5s.click(fn=lambda u: record_mic_and_transcribe(5, u),
                          inputs=[user_name_input],
                          outputs=[speech_transcribed, speech_emotion_label, speech_bars,
                                   speech_suggestion, speech_score_alert])
            rec_10s.click(fn=lambda u: record_mic_and_transcribe(10, u),
                           inputs=[user_name_input],
                           outputs=[speech_transcribed, speech_emotion_label, speech_bars,
                                    speech_suggestion, speech_score_alert])
            # Upload file
            upload_btn.click(
                fn=handle_speech_input,
                inputs=[audio_input, user_name_input],
                outputs=[speech_transcribed, speech_emotion_label, speech_bars,
                         speech_suggestion, speech_score_alert]
            )

        # ── TAB 3: WEBCAM ────────────────────────────────────────
        with gr.Tab("📷 Khuôn mặt"):
            gr.HTML(f'<div style="color:{"#27AE60" if face_available else "#E84040"};padding:8px 8px 0;font-size:13px">'
                    f'{"✅ Face model sẵn sàng" if face_available else "⚠️ Face model chưa load"}</div>')
            gr.HTML('<div style="color:#8888aa;font-size:12px;padding:4px 8px 8px">'
                    '📌 Hướng dẫn: Nhấn nút 📷 để <b>chụp snapshot</b> từ webcam (hoặc upload ảnh) → nhấn Phân tích</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    webcam_input = gr.Image(
                        label="📷 Chụp ảnh webcam hoặc Upload",
                        sources=["webcam", "upload"],
                        type="numpy",
                    )
                    webcam_btn = gr.Button("🔍 Phân tích khuôn mặt", variant="primary")

                with gr.Column(scale=3):
                    webcam_out   = gr.Image(label="📸 Kết quả nhận diện")
                    webcam_label = gr.HTML()
                    webcam_bars  = gr.HTML()

            webcam_suggestion = gr.HTML()

            webcam_btn.click(
                fn=handle_webcam_input,
                inputs=[webcam_input, user_name_input],
                outputs=[webcam_out, webcam_label, webcam_bars, webcam_suggestion]
            )

        # ── TAB 4: ĐA PHƯƠNG THỨC ───────────────────────────────
        with gr.Tab("🔀 Đa phương thức"):
            gr.HTML('<div style="color:#8888aa;font-size:13px;padding:8px">Kết hợp text + giọng nói + khuôn mặt → fusion thông minh</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    mm_text  = gr.Textbox(label="💬 Text (tùy chọn)", lines=3,
                                          placeholder="Để trống nếu chỉ dùng giọng nói...")
                    mm_audio = gr.Audio(label="🎙️ Giọng nói (tùy chọn)", type="numpy",
                                        sources=["microphone", "upload"])
                    mm_image = gr.Image(label="📷 Khuôn mặt (tùy chọn)",
                                        sources=["webcam", "upload"], type="pil")
                    mm_btn   = gr.Button("🔍 Phân tích tổng hợp", variant="primary")

                with gr.Column(scale=3):
                    mm_transcribed   = gr.Textbox(label="📝 Văn bản (speech)", interactive=False)
                    mm_emotion_label = gr.HTML()
                    mm_bars          = gr.HTML()

            mm_suggestion = gr.HTML()
            mm_score      = gr.HTML()

            mm_btn.click(
                fn=handle_multimodal,
                inputs=[mm_text, mm_audio, mm_image, user_name_input],
                outputs=[mm_transcribed, mm_emotion_label, mm_bars, mm_suggestion, mm_score]
            )

        # ── TAB 5: LỊCH SỬ ──────────────────────────────────────
        with gr.Tab("📊 Lịch sử"):
            history_btn = gr.Button("🔄 Tải lịch sử", variant="secondary")
            history_out = gr.HTML()

            history_btn.click(
                fn=get_history_html,
                inputs=[user_name_input],
                outputs=[history_out]
            )

# ══════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  MENTAL HEALTH MONITOR — UI")
    print("="*60)
    print(f"  Text model  : ✅")
    print(f"  Face model  : {'✅' if face_available else '⚠️  không tìm thấy'}")
    print(f"  Speech model: {'✅ PhoWhisper (WER ~18.7%)' if speech_available else '⚠️  không tìm thấy'}")
    print("="*60)
    #print("\n  Cài Gradio nếu chưa có: pip install gradio")
    print("  Mở trình duyệt: http://localhost:7860\n")
    """
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
    )    
    
    """

    app.launch(share=True)