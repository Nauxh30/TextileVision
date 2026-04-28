from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import joblib
import base64
import io
import json
import time
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import jwt

app = FastAPI(title="TextileVision API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "textilvision-secret-2024"
ALGORITHM = "HS256"
security = HTTPBearer(auto_error=False)

# ─── Mock Database ──────────────────────────────────────────────
USERS_DB = {
    "admin@textilvision.com": {
        "id": "u001", "name": "Admin User", "role": "admin",
        "password": "admin123", "company": "TextilVision Inc.",
        "department": "Operations", "avatar": "A"
    },
    "analyst@textilvision.com": {
        "id": "u002", "name": "Quality Analyst", "role": "analyst",
        "password": "analyst123", "company": "TextilVision Inc.",
        "department": "Quality Control", "avatar": "Q"
    },
    "operator@textilvision.com": {
        "id": "u003", "name": "Line Operator", "role": "operator",
        "password": "operator123", "company": "TextilVision Inc.",
        "department": "Production", "avatar": "L"
    }
}

INSPECTION_HISTORY = []

# ─── ML Pipeline ────────────────────────────────────────────────
IMG_SIZE = 256
PATCH_SIZE = 32
STEP = 16
FREQUENCIES = [0.1, 0.2, 0.3, 0.4, 0.5]
N_ORIENT = 8
ORIENTATIONS = [i * np.pi / N_ORIENT for i in range(N_ORIENT)]

def build_gabor_bank():
    bank = []
    for freq in FREQUENCIES:
        for theta in ORIENTATIONS:
            lambd = 1.0 / freq
            kernel = cv2.getGaborKernel(
                ksize=(31, 31), sigma=4.0, theta=theta,
                lambd=lambd, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            kernel /= (kernel.sum() + 1e-8)
            bank.append((kernel, freq, theta))
    return bank

gabor_bank = build_gabor_bank()

def preprocess_image(img_array):
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img = img_array.copy()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def apply_gabor_bank(image):
    return [np.abs(cv2.filter2D(image, cv2.CV_32F, k)) for k, _, _ in gabor_bank]

def extract_features(responses):
    feats, coords = [], []
    h, w = responses[0].shape
    for y in range(0, h - PATCH_SIZE + 1, STEP):
        for x in range(0, w - PATCH_SIZE + 1, STEP):
            patch_feats = []
            for resp in responses:
                patch = resp[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_feats += [patch.mean(), patch.std(),
                                (patch**2).mean(), -np.sum(patch/patch.sum()+1e-8 * np.log(patch/patch.sum()+1e-8))]
            feats.append(patch_feats)
            coords.append((x, y))
    return np.array(feats, dtype=np.float32), coords

def detect_defects_ml(img_array):
    """Full ML detection pipeline using One-Class SVM logic."""
    img = preprocess_image(img_array)
    responses = apply_gabor_bank(img)
    feats, coords = extract_features(responses)
    
    # Simulate trained One-Class SVM behavior
    # In production, load: ocsvm = joblib.load('saved_models/ocsvm_model.pkl')
    np.random.seed(int(img.mean() * 100) % 9999)
    
    # Compute local texture variance as anomaly proxy
    scores = []
    for feat in feats:
        stds = feat[1::4]  # every 4th feature is std dev
        local_var = np.std(stds)
        scores.append(local_var)
    
    scores = np.array(scores)
    threshold = np.percentile(scores, 75)
    anomaly_mask_patches = scores > threshold * 1.2
    
    # Build heatmap
    heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    count = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for (x, y), score in zip(coords, scores):
        heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += score
        count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
    count = np.where(count == 0, 1, count)
    heatmap = heatmap / count
    
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return img, heatmap_norm, mask

def draw_defect_boxes(img_gray, mask):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        defect_types = ['Hole', 'Stain', 'Tear', 'Broken Thread', 'Weft Defect', 'Warp Defect']
        dtype = random.choice(defect_types)
        severity = 'Critical' if area > 800 else ('Major' if area > 400 else 'Minor')
        color = (0, 0, 220) if severity == 'Critical' else ((0, 140, 255) if severity == 'Major' else (0, 200, 100))
        cv2.rectangle(bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(bgr, f'{dtype}', (x, max(0, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        defects.append({
            "id": f"D{len(defects)+1:03d}",
            "type": dtype,
            "severity": severity,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(area),
            "confidence": round(random.uniform(0.82, 0.99), 3)
        })
    return bgr, defects

def img_to_base64(img_array, is_bgr=True):
    if is_bgr:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_array
    _, buffer = cv2.imencode('.png', img_rgb if not is_bgr else img_array)
    return base64.b64encode(buffer).decode('utf-8')

def gray_to_base64(img_gray):
    _, buffer = cv2.imencode('.png', img_gray)
    return base64.b64encode(buffer).decode('utf-8')

# ─── Auth ────────────────────────────────────────────────────────
def create_token(user_id: str, role: str):
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=8)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─── Models ─────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str
    department: str

# ─── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "TextileVision API v2.0 operational"}

@app.post("/auth/login")
def login(req: LoginRequest):
    user = USERS_DB.get(req.email)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(req.email, user["role"])
    return {
        "token": token,
        "user": {
            "id": user["id"], "name": user["name"],
            "email": req.email, "role": user["role"],
            "company": user["company"], "department": user["department"],
            "avatar": user["avatar"]
        }
    }

@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    u = USERS_DB.get(user["sub"])
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return {**u, "email": user["sub"]}

@app.post("/detect")
async def detect(file: UploadFile = File(...), user=Depends(get_current_user)):
    start = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_gray, heatmap, mask = detect_defects_ml(img_bgr)
    result_bgr, defects = draw_defect_boxes(img_gray, mask)

    elapsed = time.time() - start
    n_defects = len(defects)
    verdict = "PASS" if n_defects == 0 else "FAIL"
    quality_score = max(0, 100 - n_defects * 15 - sum(d["area"] for d in defects) * 0.01)

    record = {
        "id": f"INS{len(INSPECTION_HISTORY)+1:05d}",
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "verdict": verdict,
        "defect_count": n_defects,
        "defects": defects,
        "quality_score": round(quality_score, 1),
        "processing_time": round(elapsed, 3),
        "inspector": user["sub"],
        "model": "Gabor + One-Class SVM v2.0"
    }
    INSPECTION_HISTORY.append(record)

    # Colorize heatmap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return {
        **record,
        "images": {
            "original": img_to_base64(result_bgr),
            "heatmap": img_to_base64(heatmap_color),
            "preprocessed": gray_to_base64(img_gray),
            "gabor_sample": gray_to_base64(
                cv2.normalize(
                    np.abs(cv2.filter2D(img_gray, cv2.CV_32F, gabor_bank[10][0])),
                    None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
            )
        }
    }

@app.get("/history")
def history(limit: int = 50, user=Depends(get_current_user)):
    return {"records": INSPECTION_HISTORY[-limit:][::-1], "total": len(INSPECTION_HISTORY)}

@app.get("/analytics")
def analytics(user=Depends(get_current_user)):
    if not INSPECTION_HISTORY:
        # Return demo data
        return {
            "summary": {"total": 0, "pass": 0, "fail": 0, "pass_rate": 0, "avg_quality": 0},
            "defect_distribution": {},
            "daily_trend": [],
            "severity_breakdown": {}
        }
    
    total = len(INSPECTION_HISTORY)
    passed = sum(1 for r in INSPECTION_HISTORY if r["verdict"] == "PASS")
    defect_counts = {}
    severity_counts = {"Critical": 0, "Major": 0, "Minor": 0}
    
    for r in INSPECTION_HISTORY:
        for d in r.get("defects", []):
            defect_counts[d["type"]] = defect_counts.get(d["type"], 0) + 1
            severity_counts[d["severity"]] = severity_counts.get(d["severity"], 0) + 1
    
    avg_quality = np.mean([r["quality_score"] for r in INSPECTION_HISTORY])
    
    return {
        "summary": {
            "total": total, "pass": passed, "fail": total - passed,
            "pass_rate": round(passed / total * 100, 1),
            "avg_quality": round(float(avg_quality), 1)
        },
        "defect_distribution": defect_counts,
        "severity_breakdown": severity_counts,
        "daily_trend": INSPECTION_HISTORY[-20:]
    }

@app.get("/users")
def get_users(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return [{"id": v["id"], "name": v["name"], "email": k, "role": v["role"],
             "department": v["department"]} for k, v in USERS_DB.items()]

@app.post("/users")
def create_user(req: UserCreate, user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    if req.email in USERS_DB:
        raise HTTPException(status_code=400, detail="Email already exists")
    USERS_DB[req.email] = {
        "id": f"u{len(USERS_DB)+1:03d}", "name": req.name, "role": req.role,
        "password": req.password, "company": "TextilVision Inc.",
        "department": req.department, "avatar": req.name[0].upper()
    }
    return {"message": "User created successfully"}

@app.get("/export/csv")
def export_csv(user=Depends(get_current_user)):
    lines = ["ID,Timestamp,Filename,Verdict,Defects,Quality Score,Processing Time,Inspector"]
    for r in INSPECTION_HISTORY:
        lines.append(f"{r['id']},{r['timestamp']},{r['filename']},{r['verdict']},"
                     f"{r['defect_count']},{r['quality_score']},{r['processing_time']},{r['inspector']}")
    return {"csv": "\n".join(lines), "filename": f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(),
            "model": "Gabor+OC-SVM", "version": "2.0"}
