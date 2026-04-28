# TextileVision — Fabric Defect Intelligence Platform

A full-stack, production-grade web application for real-time textile fabric defect detection using the Gabor Filter + One-Class SVM pipeline from your Colab notebook.

---

## Features

- **AI Detection** — Upload any fabric image → get defect heatmap, bounding boxes, defect classification (Hole, Stain, Tear, Broken Thread…), severity rating, and quality score
- **Real-time Results** — Sub-second inference with visual overlays
- **Auth System** — Login / Register with role-based access (Admin, Analyst, Operator)
- **Dashboard** — Live stats: pass rate, defect count, quality score, system status
- **Inspection Log** — Full history of all inspections with filtering
- **Data Analysis** — Interactive charts: defect type distribution, severity breakdown, quality trend
- **Import / Export** — Bulk image import, CSV/JSON export, REST API reference
- **Administration** — User management (add/remove users, assign roles), system settings, audit log

---

## Quick Start (2 steps)

### Step 1 — Backend (Python FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

> **No backend?** The frontend works in **Demo Mode** automatically — it simulates detection results so you can explore the full UI without any setup.

### Step 2 — Frontend

Simply open `frontend/index.html` in any browser. No build step needed.

For local serving (recommended to avoid CORS issues):
```bash
cd frontend
python -m http.server 3000
```

Then open: http://localhost:3000

---

## Demo Credentials

| Email | Password | Role |
|-------|----------|------|
| admin@textilvision.com | admin123 | Admin (full access) |
| analyst@textilvision.com | analyst123 | Analyst |
| operator@textilvision.com | operator123 | Operator |

---

## Connecting Your Trained Models

If you have trained models from the Colab notebook, place them in `backend/saved_models/`:

```
backend/
  saved_models/
    ocsvm_model.pkl       ← from: joblib.dump(ocsvm, ...)
    scaler.pkl            ← from: joblib.dump(scaler, ...)
    pca.pkl               ← from: joblib.dump(pca, ...)
    autoencoder.keras     ← from: ae.save(...)
    ae_scaler.pkl
    mm_scaler.pkl
    ae_threshold.npy
```

Then in `main.py`, replace the texture-variance anomaly scoring with:
```python
ocsvm = joblib.load('saved_models/ocsvm_model.pkl')
scaler = joblib.load('saved_models/scaler.pkl')
pca = joblib.load('saved_models/pca.pkl')
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/login | Sign in, returns JWT token |
| GET | /auth/me | Current user info |
| POST | /detect | Upload image, get full analysis |
| GET | /history | Inspection records |
| GET | /analytics | Aggregated metrics |
| GET | /export/csv | Download CSV report |
| GET | /users | List users (admin only) |
| POST | /users | Create user (admin only) |
| GET | /health | System health check |

All endpoints except /auth/login and /health require `Authorization: Bearer <token>` header.

---

## Detection Pipeline (from your Colab notebook)

```
Image Upload
    ↓
Preprocessing (CLAHE + Gaussian Blur, resize 256×256)
    ↓
Gabor Filter Bank (40 filters: 5 frequencies × 8 orientations)
    ↓
Patch Feature Extraction (32×32 patches, 16px stride)
    ↓
Features: mean, std, energy, entropy per filter → 160-dim vector
    ↓
One-Class SVM anomaly scoring
    ↓
Heatmap generation + Otsu thresholding + morphological ops
    ↓
Contour detection → Bounding boxes + defect classification
    ↓
Quality score + verdict (PASS/FAIL)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Backend | Python 3.10+ / FastAPI |
| AI/ML | OpenCV (Gabor) + scikit-learn (OC-SVM) |
| Auth | JWT (PyJWT) |
| API | REST / JSON |

---

## Deploying to Production

**Backend (e.g. Railway, Render, EC2):**
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Frontend:**
- Deploy `frontend/index.html` to Netlify, Vercel, or any static host
- Update `const API = 'http://localhost:8000'` in the HTML to your deployed backend URL

---
## Snapshorts



**Dashboard:**


<img width="1911" height="855" alt="image" src="https://github.com/user-attachments/assets/cc7afad1-7416-40f3-962f-043af40a2d37" />



**With Defect:**




<img width="1575" height="795" alt="image" src="https://github.com/user-attachments/assets/de0cac81-4430-4aec-ba7c-059441d93435" />




**Without defect:**



<img width="1584" height="663" alt="image" src="https://github.com/user-attachments/assets/a5f43cea-2727-4af5-97bb-ca3d0fab23c4" />




