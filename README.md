---
title: Mimi Voice Biometrics
emoji: 🌸🐝
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🌸 Mimi Voice Biometrics Cloud API

ECAPA-TDNN Speaker Verification API for **Jitika** & **Kalyan**.
Built with FastAPI, SpeechBrain, and PyTorch.

### Endpoints:
- `POST /api/verify`: Upload raw WAV audio to verify speaker identity (`YES` = Jitika, `NO` = Kalyan / Guest).
- `GET /health`: Healthcheck endpoint.
- `GET /`: Interactive web verification tester.
