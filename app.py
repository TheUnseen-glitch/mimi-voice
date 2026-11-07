import os
import io
import json
import torch
import torchaudio
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.inference.speaker import SpeakerRecognition

app = FastAPI(title="Mimi Voice Biometrics API")

# Enable CORS for all origins so mobile browsers on justusforever.in can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JK_PROFILE = os.path.join(BASE_DIR, "jitika_voiceprint.json")
KALYAN_PROFILE = os.path.join(BASE_DIR, "kalyan_voiceprint.json")

print("🎙️ Loading ECAPA-TDNN Speaker Recognition model...")
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/tmp/spkrec-ecapa-voxceleb"
)

# Load enrolled voiceprints
with open(JK_PROFILE, "r") as f:
    jk_data = json.load(f)
jk_emb = torch.tensor(jk_data["embedding"], dtype=torch.float32).view(1, -1)

kalyan_emb = None
if os.path.exists(KALYAN_PROFILE):
    with open(KALYAN_PROFILE, "r") as f:
        k_data = json.load(f)
    kalyan_emb = torch.tensor(k_data["embedding"], dtype=torch.float32).view(1, -1)

# Strict thresholds:
# Studio uncompressed audio gives > 0.85
# Mobile microphone compression gives ~ 0.70 - 0.75
JK_THRESHOLD = 0.70

def verify_audio(data: np.ndarray, sr: int):
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    signal = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    if sr != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)

    with torch.no_grad():
        emb = model.encode_batch(signal)
        emb = torch.nn.functional.normalize(emb, dim=-1).squeeze(1)
        sim_jk = torch.cosine_similarity(jk_emb.to(emb.device), emb, dim=-1).item()
        sim_kalyan = -1.0
        if kalyan_emb is not None:
            sim_kalyan = torch.cosine_similarity(kalyan_emb.to(emb.device), emb, dim=-1).item()

    pct_jk = max(0.0, min(100.0, sim_jk * 100))
    pct_kalyan = max(0.0, min(100.0, sim_kalyan * 100))

    # Dual verification logic:
    # 1. Must beat threshold
    # 2. Must score significantly higher than Kalyan to prevent couple cross-confusion
    is_jitika = sim_jk >= JK_THRESHOLD and sim_jk > (sim_kalyan + 0.10)
    is_kalyan = sim_kalyan >= 0.55 and sim_kalyan > (sim_jk + 0.10)

    if is_jitika:
        speaker = "jitika"
        verdict = "YES"
        title = "YES ✨"
        subtitle = "JITIKA'S VOICE RECOGNIZED 🌸"
        color = "#10b981"
    elif is_kalyan:
        speaker = "kalyan"
        verdict = "NO"
        title = "NO 👑"
        subtitle = "KALYAN'S VOICE DETECTED"
        color = "#3b82f6"
    else:
        speaker = "guest"
        verdict = "NO"
        title = "NO ❌"
        subtitle = "NOT JITIKA (Another Voice / Guest)"
        color = "#ef4444"

    return {
        "verdict": verdict,
        "is_jitika": is_jitika,
        "speaker": speaker,
        "similarity_jitika": round(sim_jk, 4),
        "similarity_kalyan": round(sim_kalyan, 4),
        "percentage": f"{pct_jk:.1f}%",
        "title": title,
        "subtitle": subtitle,
        "color": color
    }

@app.get("/health")
def health():
    return {"status": "online", "model": "ECAPA-TDNN", "speaker_enrolled": "Jitika"}

@app.post("/api/verify")
async def verify(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart" in content_type:
        form = await request.form()
        audio_file = form.get("audio") or form.get("file")
        if not audio_file:
            return JSONResponse({"error": "No audio file provided in form-data"}, status_code=400)
        audio_bytes = await audio_file.read()
    else:
        audio_bytes = await request.body()

    if not audio_bytes or len(audio_bytes) < 500:
        return JSONResponse({"error": "Audio clip too short or empty"}, status_code=400)

    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        result = verify_audio(data, sr)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": f"Audio processing failed: {str(e)}"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mimi Voice Biometrics Cloud API</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f0a1c; color: #fff; padding: 2rem; text-align: center; }
    .card { background: #1c1328; border: 1px solid #38234a; border-radius: 16px; max-width: 500px; margin: 2rem auto; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    h1 { color: #fbbf24; font-size: 1.6rem; }
    .btn { background: linear-gradient(135deg, #f43f5e, #e11d48); color: #fff; border: none; padding: 12px 24px; border-radius: 999px; font-weight: bold; cursor: pointer; margin: 10px; }
    .status { margin-top: 1.5rem; padding: 1rem; border-radius: 12px; font-size: 1.2rem; font-weight: bold; display: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🌸 Mimi Biometrics Cloud API</h1>
    <p style="color: #fda4af;">ECAPA-TDNN Neural Voice Recognition for Jitika</p>
    <div style="margin: 2rem 0;">
      <button class="btn" id="recordBtn" onclick="toggleRecord()">🎙️ Test Live Mic</button>
    </div>
    <div id="statusBox" class="status"></div>
  </div>
  <script>
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    async function toggleRecord() {
      const btn = document.getElementById('recordBtn');
      const box = document.getElementById('statusBox');
      if (!isRecording) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
          btn.textContent = "⏳ Verifying Voice...";
          const blob = new Blob(audioChunks, { type: 'audio/wav' });
          const res = await fetch('/api/verify', { method: 'POST', body: blob });
          const data = await res.json();
          box.style.display = 'block';
          box.style.backgroundColor = data.color + '22';
          box.style.border = '2px solid ' + data.color;
          box.style.color = data.color;
          box.innerHTML = '<div>' + data.title + '</div><div style="font-size:0.9rem; margin-top:4px;">' + data.subtitle + ' (' + data.percentage + ')</div>';
          btn.textContent = "🎙️ Test Live Mic";
        };
        mediaRecorder.start();
        isRecording = true;
        btn.textContent = "⏹️ Stop & Verify";
        box.style.display = 'none';
      } else {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());
        isRecording = false;
      }
    }
  </script>
</body>
</html>"""
