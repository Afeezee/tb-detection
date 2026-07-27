# TB Detection — Deep Learning Model for Early Detection of Tuberculosis

Student: Adesanlu Martins (U/22/CS/0011) · Supervisor: Miss Shadare

Chest X-ray screening for pulmonary tuberculosis using a DenseNet121 baseline
and a Hybrid CNN+ViT novelty model. Deployed as a two-service monorepo on
Railway: a FastAPI backend that wraps the trained PyTorch models, and a
Next.js frontend for the clinical UI.

## Repo layout

```
tb-detection/
├── backend/           FastAPI service (wraps src/ modules)
├── frontend/          Next.js 14 (App Router) UI
├── models/            densenet121_best.pt, hybrid_best.pt
├── src/               training-time modules (untouched)
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── db.py
├── config.py          project-wide config (paths, hyperparameters)
├── app.py             legacy Streamlit demo (kept for reference)
├── railway.toml       root marker for the monorepo
├── .env.example       template of every required env var
└── README.md
```

## 1. Trained models

Two checkpoints ship in `models/`:

| Model              | Regime        | Test set          | Sensitivity | Specificity | F1     | AUC-ROC |
| ------------------ | ------------- | ----------------- | ----------- | ----------- | ------ | ------- |
| DenseNet121        | multi-source  | internal          | 0.9911      | 0.9842      | 0.9407 | 0.9987  |
| Hybrid CNN+ViT     | multi-source  | internal          | 0.9911      | 0.9812      | 0.9308 | 0.9980  |
| DenseNet121 (ref.) | single-source | external TBX11K   | —           | —           | —      | 0.5581  |

Multi-source training was adopted after the single-source baseline collapsed to
near-chance AUC on the external TBX11K split — evidence of shortcut learning
now avoided.

## 2. Local development

### Backend

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
copy .env.example .env           # then fill in NEON_DATABASE_URL
python -m src.db                 # one-off: creates the tb_predictions table
uvicorn main:app --app-dir backend --reload --port 8000
```

Sanity check: `http://localhost:8000/health` should return
`{"status":"ok", ...}` and `http://localhost:8000/docs` gives the OpenAPI UI.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local     # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

Then open `http://localhost:3000/predict`.

## 3. API surface

| Method | Path                     | Purpose                                                  |
| ------ | ------------------------ | -------------------------------------------------------- |
| POST   | `/predict?model=…`       | Classify an uploaded chest X-ray; returns label,          |
|        |                          | confidence, base64 Grad-CAM PNG, model_used. Persists     |
|        |                          | to Neon by default (set `persist=false` to skip).         |
| GET    | `/history?limit=50`      | Recent predictions from Neon.                             |
| GET    | `/metrics`               | Hardcoded benchmark table for both models.                |
| GET    | `/health`                | Health probe (used by Railway).                           |

Model switcher: `POST /predict?model=densenet121` (default, faster) or
`?model=hybrid`.

## 4. Deploy to Railway

The repo deploys as **one Railway project containing two services** —
`backend` and `frontend`.

Prerequisites: `railway login` completed, project pushed to GitHub (Railway
prefers GitHub-linked services; the CLI can also stream a local tarball via
`railway up`).

### 4.1. One-time project setup

```bash
railway init                     # create a new Railway project, or `railway link` an existing one
railway add                      # add a Postgres plugin only if not using Neon; skip for Neon
```

### 4.2. Create the backend service

Set the service's **Root Directory** to `backend` and its **Config Path** to
`backend/railway.toml`. If using the CLI:

```bash
railway service create backend
railway service backend
railway variables set \
  NEON_DATABASE_URL="postgresql://..." \
  DEFAULT_MODEL="densenet121" \
  MODEL_DIR="/app/models"
railway up                       # from the repo root; Dockerfile at backend/Dockerfile
```

The Dockerfile builds with the **repo root** as its build context so it can
`COPY config.py`, `src/`, and `models/` alongside the backend code. Railway
picks this up from `backend/railway.toml` (`dockerfilePath = "backend/Dockerfile"`).

Note the resulting public URL — call it `BACKEND_URL` below.

### 4.3. Create the frontend service

```bash
railway service create frontend
railway service frontend
railway variables set NEXT_PUBLIC_API_BASE_URL="$BACKEND_URL"
railway up
```

Note the frontend public URL — call it `FRONTEND_URL`.

### 4.4. Wire up CORS

Add `FRONTEND_URL` to the backend service so its CORS allow-list accepts the
Next.js origin:

```bash
railway service backend
railway variables set FRONTEND_URL="$FRONTEND_URL"
railway redeploy
```

### 4.5. Post-deploy checks

- `GET $BACKEND_URL/health` returns 200 with `default_model` and `loaded_models`.
- `GET $BACKEND_URL/metrics` returns the four benchmark rows.
- `$FRONTEND_URL/predict` uploads an image and shows the Grad-CAM overlay.
- `$FRONTEND_URL/history` lists the saved row.

## 5. Environment variables

Everything the app needs, documented in `.env.example`. Set each on the
matching Railway service:

| Variable                    | Where       | Purpose                                                        |
| --------------------------- | ----------- | -------------------------------------------------------------- |
| `NEON_DATABASE_URL`         | backend     | Neon Postgres connection string.                               |
| `MODEL_DIR`                 | backend     | Container path to `.pt` checkpoints (default `/app/models`).    |
| `DEFAULT_MODEL`             | backend     | `densenet121` (recommended default) or `hybrid`.                |
| `FRONTEND_URL`              | backend     | Public URL of the frontend, for the CORS allow-list.            |
| `NEXT_PUBLIC_API_BASE_URL`  | frontend    | Public URL of the backend, baked in at `next build`.            |
| `PORT`                      | both        | Injected automatically by Railway — do not set manually.        |

## 6. Legacy Streamlit demo

`app.py` at the repo root is the original Streamlit interface. It still works
against the same `src/` modules and is kept as an offline fallback. Run with
`streamlit run app.py`.

## 7. What's next

- Drop the two ROC curve PNGs into `frontend/public/roc/` as
  `densenet121_roc.png` and `hybrid_roc.png` so the /benchmark page can render
  them.
- Optional: attention-rollout for the ViT branch to complement Grad-CAM on the
  hybrid model.

---

Research prototype only — not a licensed medical device.
