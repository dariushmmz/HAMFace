# HAMFace Dashboard

FastAPI web dashboard for the HAMFace recognition/authentication system.

## Directory layout

```
face_recognition/          ← your existing project root
└── dashboard/
    ├── app.py             ← FastAPI application
    ├── requirements.txt
    └── templates/
        ├── index.html     ← recognition dashboard
        └── enroll.html    ← enroll new person page
```

## Setup

```bash
cd face_recognition/dashboard
pip install -r requirements.txt
```

## Run

```bash
# from face_recognition/ root (so imports like `from config import …` resolve)
uvicorn dashboard.app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000**

## Pages

| URL | Description |
|-----|-------------|
| `/` | Live dashboard — image / video / webcam recognition |
| `/enroll` | Add a new person to the gallery |
| `/api/status` | JSON — gallery size, known persons, model state |
| `/api/recognize/image` | POST an image file, returns annotated JPEG + results |
| `/api/recognize/video_frame` | POST a JPEG frame, same response |
| `/api/enroll/person` | POST name + images to add to gallery |
| `/api/persons` | GET list of all enrolled persons |
| `/ws/webcam` | WebSocket — send JPEG frames, receive annotated frames |

## Notes

- **Model weights** must exist at `checkpoints/hamface_model.h5` before the dashboard
  can run inference. Train with `python train.py` first.
- **Gallery** is loaded lazily on first request and cached in memory.
- Enrolling a new person via the UI updates `gallery_avg.pkl` **in-place** without
  retraining — new embeddings are averaged with any existing ones for that person.
- The webcam stream runs over WebSocket at ~2 fps by default (configurable in JS).
