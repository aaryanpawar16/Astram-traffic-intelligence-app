# 🚦 ASTRAM Event Congestion Intelligence Platform

> **AI-powered traffic management system** that forecasts event-driven congestion, recommends optimal manpower & barricading, and generates diversion plans — built for the Smart Traffic Challenge.

---

## 📸 Features at a Glance

| Tab | What it does |
|-----|-------------|
| 📊 **Overview** | Live Congestion Risk Gauge (0–100) + KPI cards + 4 analytical charts |
| ⏱ **Temporal Patterns** | Hourly distribution, day-of-week breakdown, Zone × Hour heatmap |
| ⚠️ **Risk Matrix** | Quantified risk scores per event type with radar chart |
| 🤖 **AI Event Planner** | GPT-4o deployment plan + 3 similar past incidents + PDF report download |
| 📈 **Impact Simulator** | Before vs After ASTRAM — projected closure rate & time reduction |
| 🗺 **Diversion Playbook** | Data-derived SOPs + post-event learning library |

---

## 🗂 Project Structure

```
ASTRAM/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .env                    ← Your API key (never commit this!)
├── .env.example            ← Template for .env
├── README.md               ← This file
└── ASTRAM.csv / .xlsx      ← Event dataset (place here)
```

---

## ⚡ Quick Start

### 1. Clone / download the project
```bash
git clone https://github.com/your-username/astram-traffic.git
cd astram-traffic
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the project folder:
```
OPENAI_API_KEY=sk-your-openai-key-here
```

### 4. Place the dataset
Put your `ASTRAM.csv` (or `.xlsx`) file in the same folder as `app.py`.  
The app auto-detects it — no renaming needed.

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key — get one at [platform.openai.com](https://platform.openai.com) |

> ⚠️ **Never commit your `.env` file to GitHub.** Add it to `.gitignore`.

`.gitignore` should contain:
```
.env
__pycache__/
*.pyc
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.32 | Web app framework |
| `pandas` | ≥2.0 | Data processing |
| `plotly` | ≥5.18 | Interactive charts |
| `requests` | ≥2.31 | OpenAI API calls |
| `scikit-learn` | ≥1.3 | Cosine similarity for event matching |
| `numpy` | ≥1.24 | Numerical operations |
| `fpdf2` | ≥2.7 | PDF report generation |
| `python-dotenv` | ≥1.0 | Load `.env` file |
| `openpyxl` | ≥3.1 | Read `.xlsx` dataset files |

---

## 🚀 Deploy to Streamlit Community Cloud (Free)

1. Push your code to a **public GitHub repo** (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set `app.py` as the main file
4. Go to **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy** — live in ~2 minutes ✅

---

## 🧠 How the AI Planner Works

1. You fill in event type, zone, corridor, time, and crowd size
2. The app runs **two things in parallel**:
   - Calls **GPT-4o** with a system prompt containing historical statistics from 8,173 ASTRAM incidents
   - Runs a **cosine similarity search** against the dataset to find the 3 most similar past events
3. GPT-4o returns a structured 4-section plan:
   - 📊 Impact Forecast (closure probability, radius, duration)
   - 👮 Manpower Deployment Plan (exact units, positions, timings)
   - 🚧 Barricading Strategy (sets, placement, VMS distances)
   - 🗺️ Diversion Routes (primary + secondary with travel time)
4. A **PDF report** is generated with all of the above + historical evidence table

---

## 📊 Dataset

The app expects the ASTRAM event dataset with these key columns:

| Column | Description |
|--------|-------------|
| `event_type` | `planned` or `unplanned` |
| `event_cause` | `vehicle_breakdown`, `public_event`, `vip_movement`, etc. |
| `zone` | City zone (e.g. `Central Zone 1`) |
| `corridor` | Road corridor name |
| `start_datetime` | Event start timestamp |
| `closed_datetime` | Event resolution timestamp |
| `requires_road_closure` | Boolean — whether road was closed |
| `priority` | `High`, `Medium`, or `Low` |
| `hour` | *(computed)* Hour of day |
| `duration_min` | *(computed)* Resolution time in minutes |

---

## 🏆 Built For

**Smart Traffic Challenge** — Problem Statement: *Event-Driven Congestion (Planned & Unplanned)*

Addresses all 3 stated gaps:
- ✅ **Event impact quantified in advance** → Risk Matrix + AI Impact Forecast
- ✅ **Data-driven resource deployment** → AI Planner with historical grounding
- ✅ **Post-event learning system** → Diversion Playbook + Similar Incidents engine

---

## 👤 Author
Aaryan Pawar, Utkarsh Pandey
Built with ❤️ using Streamlit, GPT-4o, and the ASTRAM dataset.