# LendVerify - Quick Start Guide

## 🚀 What Was Fixed

### 1. **Dark Theme Applied** ✅
- Changed background from light blue to dark navy (`#0f172a`)
- Updated all UI elements (cards, text, buttons) for dark mode
- Fixed file paths to your actual workspace location

### 2. **Backend-Frontend Connection Fixed** ✅
**Problem:** The frontend was calling local Python functions instead of the FastAPI backend
**Solution:** 
- Now ALL operations go through API calls (`/api/v1/assess`, `/api/v1/verify`, `/api/v1/ml/predict`)
- Added detailed logging to track API requests

### 3. **Added Comprehensive Logging** ✅
- Frontend logs every API call (check console for `INFO` messages)
- Backend logs every request to `api.log` file AND console
- Now you can see EXACTLY what's happening when errors occur

---

## 🏃 How to Run

### Step 1: Start the Backend (Terminal 1)
```bash
cd /Users/sukrit/.hermes/profiles/personal-ai-academy/SMT
python main.py
```

**You should see:**
```
INFO:     Started server process [PID]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Health check requested
```

The backend will log EVERY request to `api.log` and the terminal.

### Step 2: Start the Frontend (Terminal 2)
```bash
streamlit run app.py
```

This opens the dark-themed UI in your browser.

---

## 🔍 How to Debug Issues

### If "Cannot connect to backend" appears:

1. **Check if backend is running:**
   ```bash
   curl http://127.0.0.1:8000/api/v1/health
   ```
   Should return: `{"status":"ok","service":"lendverify-api"}`

2. **Check backend logs:**
   - Look at the terminal running `python main.py`
   - OR check the log file: `cat api.log`

3. **Check if port 8000 is free:**
   ```bash
   lsof -i :8000
   # If it shows processes, kill them first:
   kill <PID>
   ```

### To see what API calls are happening:

**Frontend logs (in Streamlit console):**
```python
logger.info("Sending assessment request to backend")
logger.info("Sending verification request to backend: defect=<type>")
```

**Backend logs (`api.log` file):**
```
2026-10-23 10:30:45 | INFO | Assessment request: {...}
2026-10-23 10:30:46 | INFO | Verification request: defect=profitability
```

---

## 📋 API Endpoints Available

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Check if backend is running |
| `/api/v1/assess` | POST | Assess borrower policies |
| `/api/v1/verify` | POST | Run Z3 SMT verification |
| `/api/v1/ml/status` | GET | Get model status |
| `/api/v1/ml/train` | POST | Train ML models |
| `/api/v1/ml/predict` | POST | Predict loan outcome |
| `/api/v1/ml/evaluate-test` | POST | Evaluate test set |

---

## ⚠️ Common Issues & Fixes

### Issue: "ConnectionError: Cannot connect to backend"
**Fix:** Start `python main.py` first in a separate terminal before opening Streamlit

### Issue: "Timeout" on API calls  
**Fix:** Check if Z3 is working (it can be slow), increase timeout if needed

### Issue: Backend returns errors
**Fix:** Check the backend logs - they will show the exact error message and stack trace

---

## 🎨 Theme Colors

- **Background:** `#0f172a` (Dark Navy)
- **Cards:** `#1e293b` (Lighter Navy)  
- **Text:** `#e2e8f0` (Light Gray)
- **Accent Blue:** `#3b82f6` (Bright Blue)
