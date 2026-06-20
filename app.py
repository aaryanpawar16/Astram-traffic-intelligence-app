import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests, os, io, re
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fpdf import FPDF

# Load API key from .env
load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ASTRAM | Event Congestion Intelligence",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1e2d4a; border-radius: 12px;
    padding: 20px; text-align: center; margin-bottom: 8px;
  }
  .metric-val   { font-size: 2rem; font-weight: 700; }
  .metric-label { font-size: 0.7rem; color: #64748b; letter-spacing: 1.5px; text-transform: uppercase; font-family: 'JetBrains Mono'; }
  .metric-sub   { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
  .section-badge {
    display: inline-block; background: rgba(0,229,255,0.1); color: #00e5ff;
    border: 1px solid rgba(0,229,255,0.2); border-radius: 4px;
    padding: 2px 10px; font-size: 0.7rem; letter-spacing: 1.5px; font-family: 'JetBrains Mono';
  }
  .ai-block {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1e2d4a; border-radius: 10px; padding: 16px; margin-bottom: 12px;
  }
  .ai-block h4 { color: #00e5ff; font-family: 'JetBrains Mono'; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 8px; }
  .playbook-card {
    background: linear-gradient(135deg, #111827, #1a2540);
    border-left: 3px solid #7c3aed; border-radius: 0 10px 10px 0;
    padding: 16px; margin-bottom: 12px;
  }
  .similar-card {
    background: linear-gradient(135deg, #0f1f35, #111827);
    border: 1px solid #1e2d4a; border-left: 3px solid #10b981;
    border-radius: 0 10px 10px 0; padding: 14px; margin-bottom: 10px;
  }
  .risk-gauge-label { font-size: 0.7rem; color: #64748b; letter-spacing: 1px; font-family: 'JetBrains Mono'; }
  div[data-testid="stSidebar"] { background: #0d1526; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: #111827; border-radius: 6px; color: #64748b; }
  .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7c3aed, #00e5ff); color: white; }
</style>
""", unsafe_allow_html=True)

# ── FIND DATA FILE ────────────────────────────────────────────────
def find_csv():
    search_dirs = [Path("."), Path(__file__).parent]
    patterns = ["Astram_event_data*","ASTRAM*.csv","ASTRAM*.xlsx","ASTRAM*.xls",
                "astram*.csv","astram*.xlsx","*.csv"]
    for d in search_dirs:
        for pat in patterns:
            matches = list(d.glob(pat))
            if matches: return str(matches[0])
    return None

# ── LOAD DATA ─────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path) if path.endswith((".xlsx",".xls")) else pd.read_csv(path)
    df["start_datetime"]  = pd.to_datetime(df["start_datetime"],  format="mixed", utc=True)
    df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], format="mixed", utc=True, errors="coerce")
    df["duration_min"] = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
    df["hour"]        = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.day_name()
    df["month"]       = df["start_datetime"].dt.month_name()
    return df

csv_path = find_csv()
if csv_path is None:
    st.error("❌ CSV file not found. Place the ASTRAM CSV in the same folder as app.py.")
    st.stop()
df = load_data(csv_path)

# ── PLOTLY THEME ──────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e2d4a", showline=False),
    yaxis=dict(gridcolor="#1e2d4a", showline=False),
    margin=dict(l=10, r=10, t=30, b=10),
)
PAL = ["#00e5ff","#7c3aed","#ff6b35","#10b981","#f59e0b","#ef4444","#06b6d4","#a78bfa","#fb923c","#34d399"]

RISK_DATA = [
    {"Cause":"VIP Movement",     "Closure %":80,"Duration Score":50,"Manpower Score":95,"Police Units":8, "Barricades":12,"Priority":"🔴 HIGH"},
    {"Cause":"Public Event",     "Closure %":46,"Duration Score":30,"Manpower Score":70,"Police Units":5, "Barricades":8, "Priority":"🔴 HIGH"},
    {"Cause":"Protest / Bandh",  "Closure %":40,"Duration Score":20,"Manpower Score":65,"Police Units":6, "Barricades":6, "Priority":"🔴 HIGH"},
    {"Cause":"Tree Fall",        "Closure %":39,"Duration Score":60,"Manpower Score":45,"Police Units":2, "Barricades":4, "Priority":"🟡 MEDIUM"},
    {"Cause":"Construction",     "Closure %":27,"Duration Score":90,"Manpower Score":60,"Police Units":3, "Barricades":10,"Priority":"🟡 MEDIUM"},
    {"Cause":"Procession",       "Cause2":"procession","Closure %":26,"Duration Score":25,"Manpower Score":50,"Police Units":4,"Barricades":5,"Priority":"🟡 MEDIUM"},
    {"Cause":"Road Conditions",  "Closure %":12,"Duration Score":85,"Manpower Score":30,"Police Units":1, "Barricades":3, "Priority":"🟢 LOW"},
    {"Cause":"Water Logging",    "Closure %": 9,"Duration Score":75,"Manpower Score":35,"Police Units":2, "Barricades":4, "Priority":"🟢 LOW"},
    {"Cause":"Congestion",       "Closure %": 4,"Duration Score":15,"Manpower Score":25,"Police Units":2, "Barricades":2, "Priority":"🟢 LOW"},
    {"Cause":"Vehicle Breakdown","Closure %": 4,"Duration Score":12,"Manpower Score":20,"Police Units":1, "Barricades":1, "Priority":"🟢 LOW"},
    {"Cause":"Accident",         "Closure %": 3,"Duration Score":12,"Manpower Score":30,"Police Units":3, "Barricades":3, "Priority":"🟢 LOW"},
    {"Cause":"Pot Holes",        "Closure %": 2,"Duration Score":80,"Manpower Score":15,"Police Units":0, "Barricades":2, "Priority":"🟢 LOW"},
]

# ── OPENAI HELPER ─────────────────────────────────────────────────
def call_openai(prompt: str, system: str) -> str:
    if not OPENAI_API_KEY:
        return "❌ OPENAI_API_KEY not found in .env file."
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model":"gpt-4o","messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
                  "max_tokens":1200,"temperature":0.4},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        if resp.status_code == 401: return "❌ Invalid API key."
        if resp.status_code == 429: return "❌ Rate limit exceeded."
        return f"❌ HTTP {resp.status_code}"
    except Exception as e:
        return f"❌ Error: {e}"

# ════════════════════════════════════════════════════════════════════
# FEATURE A — TODAY'S RISK INDEX
# ════════════════════════════════════════════════════════════════════
def compute_today_risk(df):
    now = datetime.now()
    hour = now.hour
    dow  = now.strftime("%A")
    # Hour weight (from data: peak 20-22h, early 4-7h)
    hour_weights = {
        range(20,23):1.0, range(4,8):0.85, range(18,20):0.75,
        range(7,10):0.65, range(0,4):0.5,  range(23,24):0.5,
        range(10,18):0.3,
    }
    hw = 0.3
    for rng, w in hour_weights.items():
        if hour in rng: hw = w; break
    # Day weight
    day_w = {"Friday":1.0,"Thursday":0.95,"Tuesday":0.92,"Saturday":0.90,
              "Wednesday":0.88,"Sunday":0.72,"Monday":0.68}.get(dow, 0.8)
    # Month weight
    month_events = df["month"].value_counts()
    month_name = now.strftime("%B")
    month_max = month_events.max() if len(month_events) else 1
    month_w = month_events.get(month_name, month_events.mean()) / month_max

    score = round((hw*0.45 + day_w*0.35 + float(month_w)*0.20) * 100)
    score = min(max(score, 5), 98)

    if score >= 75:   level, color = "CRITICAL", "#ef4444"
    elif score >= 55: level, color = "HIGH",     "#f59e0b"
    elif score >= 35: level, color = "MEDIUM",   "#00e5ff"
    else:             level, color = "LOW",       "#10b981"
    return score, level, color, hour, dow

# ════════════════════════════════════════════════════════════════════
# FEATURE B — SIMILAR EVENT FINDER
# ════════════════════════════════════════════════════════════════════
@st.cache_data
def build_similarity_index(df):
    """Encode categorical columns for cosine similarity search."""
    cols = ["event_cause","zone","corridor","hour","day_of_week"]
    sub = df[cols + ["requires_road_closure","duration_min","priority"]].dropna(subset=cols)
    encoded = sub[cols].copy()
    encoders = {}
    for c in ["event_cause","zone","corridor","day_of_week"]:
        le = LabelEncoder()
        encoded[c] = le.fit_transform(sub[c].astype(str))
        encoders[c] = le
    return sub.reset_index(drop=True), encoded.astype(float).values, encoders

def find_similar_events(df, event_type, zone, corridor, time_slot, n=3):
    cause_map = {
        "Political Rally":"public_event","Religious Procession":"procession",
        "Sports Match":"public_event","Music Festival":"public_event",
        "VIP Movement":"vip_movement","Protest / Bandh":"protest",
        "Cultural Festival":"public_event","Marathon / Run":"procession",
        "Road Construction":"construction",
    }
    time_hour_map = {
        "Early Morning (4–7 AM)":5,"Morning Peak (7–10 AM)":8,
        "Midday (10 AM–2 PM)":12,"Evening Peak (5–8 PM)":18,
        "Night (8 PM–12 AM)":20,"Late Night (12–4 AM)":2,
    }
    cause = cause_map.get(event_type, "public_event")
    hour  = time_hour_map.get(time_slot, 12)

    sub, matrix, encoders = build_similarity_index(df)

    # Build query vector
    q = {}
    try: q["event_cause"] = encoders["event_cause"].transform([cause])[0]
    except: q["event_cause"] = 0
    try: q["zone"] = encoders["zone"].transform([zone])[0]
    except: q["zone"] = 0
    try: q["corridor"] = encoders["corridor"].transform([corridor or "Non-corridor"])[0]
    except: q["corridor"] = 0
    q["hour"] = hour
    try: q["day_of_week"] = encoders["day_of_week"].transform([datetime.now().strftime("%A")])[0]
    except: q["day_of_week"] = 0

    query_vec = np.array([[q["event_cause"], q["zone"], q["corridor"], q["hour"], q["day_of_week"]]])
    sims = cosine_similarity(query_vec, matrix)[0]
    top_idx = sims.argsort()[-n:][::-1]
    return sub.iloc[top_idx].copy(), sims[top_idx]

# ════════════════════════════════════════════════════════════════════
# FEATURE C — BEFORE/AFTER IMPACT SIMULATION
# ════════════════════════════════════════════════════════════════════
def compute_impact_simulation(df, event_type):
    cause_map = {
        "Political Rally":"public_event","Religious Procession":"procession",
        "Sports Match":"public_event","Music Festival":"public_event",
        "VIP Movement":"vip_movement","Protest / Bandh":"protest",
        "Cultural Festival":"public_event","Marathon / Run":"procession",
        "Road Construction":"construction","":"public_event",
    }
    cause = cause_map.get(event_type, "public_event")
    subset = df[df["event_cause"] == cause]

    fallback_durations = {
        "public_event": 49, "vip_movement": 22, "protest": 31,
        "procession": 37,   "construction": 2964,
    }

    import math
    base_closure  = subset["requires_road_closure"].mean() * 100
    raw_dur       = subset["duration_min"].dropna()
    base_duration = float(raw_dur.median()) if len(raw_dur) >= 3 else float(fallback_durations.get(cause, 60))

    if math.isnan(base_closure):  base_closure  = 10.0
    if math.isnan(base_duration): base_duration = float(fallback_durations.get(cause, 60))

    improvements = {
        "public_event": {"closure_red": 0.38, "duration_red": 0.31},
        "vip_movement": {"closure_red": 0.15, "duration_red": 0.20},
        "protest":      {"closure_red": 0.30, "duration_red": 0.25},
        "procession":   {"closure_red": 0.35, "duration_red": 0.28},
        "construction": {"closure_red": 0.42, "duration_red": 0.20},
    }
    imp           = improvements.get(cause, {"closure_red": 0.30, "duration_red": 0.25})
    proj_closure  = base_closure  * (1 - imp["closure_red"])
    proj_duration = base_duration * (1 - imp["duration_red"])

    return {
        "base_closure":  round(base_closure,  1),
        "proj_closure":  round(proj_closure,  1),
        "base_duration": round(base_duration, 1),
        "proj_duration": round(proj_duration, 1),
        "closure_saved": round(base_closure  - proj_closure,  1),
        "time_saved":    round(base_duration - proj_duration, 1),
        "closure_pct":   round(imp["closure_red"]  * 100),
        "duration_pct":  round(imp["duration_red"] * 100),
        "n_events":      len(subset),
    }

# ════════════════════════════════════════════════════════════════════
# FEATURE D — PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════
def sanitize(text):
    """Strip all non-Latin-1 characters so fpdf Helvetica never chokes."""
    import unicodedata
    # Common Unicode replacements first
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "•": "*", "→": "->",
        "✓": "OK", "✔": "OK", "❌": "X",  "⚠": "!",
        "✅": "OK", "❤": "<3", "➤": ">",
        "é": "e", "è": "e", "ê": "e",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Drop anything still outside Latin-1 (0x00-0xFF)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def generate_pdf(event_type, zone, corridor, time_slot, crowd_size, ai_plan, similar_events_df, sim_data):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(10, 15, 30)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(0, 180, 220)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "ASTRAM Event Deployment Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.set_x(10)
    pdf.cell(0, 6, sanitize(
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}  |  Event Congestion Intelligence Platform"
    ), ln=True)

    # Event Summary Box
    pdf.set_fill_color(17, 24, 39)
    pdf.set_draw_color(30, 45, 74)
    pdf.rect(10, 45, 190, 38, "FD")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 180, 220)
    pdf.set_xy(14, 48)
    pdf.cell(0, 7, "EVENT DETAILS", ln=True)
    details = [
        ("Event Type", event_type or "N/A"),
        ("Zone",       zone        or "N/A"),
        ("Corridor",   corridor    or "N/A"),
        ("Time",       time_slot   or "N/A"),
        ("Crowd Size", crowd_size  or "N/A"),
    ]
    col_w = 90
    for i, (k, v) in enumerate(details):
        x = 14 + (i % 2) * col_w
        y = 56 + (i // 2) * 8
        pdf.set_xy(x, y)
        pdf.set_text_color(100, 116, 139)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(32, 5, sanitize(k.upper() + ":"), ln=False)
        pdf.set_text_color(226, 232, 240)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, sanitize(str(v)), ln=False)

    # Impact Simulation
    pdf.set_fill_color(17, 24, 39)
    pdf.rect(10, 88, 190, 40, "FD")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 180, 220)
    pdf.set_xy(14, 91)
    pdf.cell(0, 7, "IMPACT SIMULATION -- WITH vs WITHOUT ASTRAM", ln=True)
    metrics_sim = [
        ("Road Closure Rate (Before)", f"{sim_data['base_closure']}%"),
        ("Road Closure Rate (After)",  f"{sim_data['proj_closure']}% (-{sim_data['closure_pct']}%)"),
        ("Avg Duration (Before)",      f"{sim_data['base_duration']} min"),
        ("Avg Duration (After)",       f"{sim_data['proj_duration']} min (-{sim_data['duration_pct']}%)"),
    ]
    for i, (k, v) in enumerate(metrics_sim):
        x = 14 + (i % 2) * 95
        y = 101 + (i // 2) * 10
        pdf.set_xy(x, y)
        pdf.set_text_color(100, 116, 139)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(55, 5, sanitize(k + ":"), ln=False)
        color = (16, 185, 129) if "After" in k else (239, 68, 68)
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, sanitize(v), ln=False)

    # AI Plan
    pdf.set_y(136)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 180, 220)
    pdf.cell(0, 8, "AI-GENERATED DEPLOYMENT PLAN (GPT-4o)", ln=True)
    pdf.set_draw_color(0, 180, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Clean markdown + unicode from AI plan
    clean_plan = re.sub(r"#{1,4}\s*", "", ai_plan)
    clean_plan = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", clean_plan)
    clean_plan = sanitize(clean_plan)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(190, 5, clean_plan)

    # Similar Events Table
    pdf.ln(4)
    if pdf.get_y() > 240:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 180, 220)
    pdf.cell(0, 8, "3 MOST SIMILAR HISTORICAL INCIDENTS", ln=True)
    pdf.set_draw_color(0, 180, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    headers    = ["Cause", "Zone", "Corridor", "Closed?", "Duration(min)", "Priority"]
    col_widths = [32, 30, 42, 18, 28, 20]
    pdf.set_fill_color(26, 37, 64)
    pdf.set_text_color(0, 180, 220)
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for _, row in similar_events_df.iterrows():
        vals = [
            sanitize(str(row.get("event_cause", ""))[:16].replace("_", " ")),
            sanitize(str(row.get("zone", ""))[:16]),
            sanitize(str(row.get("corridor", ""))[:20]),
            "YES" if row.get("requires_road_closure") else "NO",
            f"{row.get('duration_min', 0):.0f}" if pd.notna(row.get("duration_min")) else "N/A",
            sanitize(str(row.get("priority", ""))[:8]),
        ]
        pdf.set_text_color(30, 30, 30)
        for v, w in zip(vals, col_widths):
            pdf.cell(w, 6, v, border=1)
        pdf.ln()

    # Footer
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "ASTRAM Event Congestion Intelligence Platform  |  Confidential -- For Operational Use Only", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 ASTRAM")
    st.markdown("<span class='section-badge'>EVENT CONGESTION INTELLIGENCE</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📂 Data")
    st.success(f"✅ **{len(df):,}** events loaded")
    st.caption(f"`{Path(csv_path).name}`")
    st.markdown("---")
    st.markdown("### 🔑 OpenAI API Key")
    if OPENAI_API_KEY:
        st.success("GPT-4o ready ✓")
    else:
        st.error("Add OPENAI_API_KEY to .env file")
    st.markdown("---")
    st.markdown("### 🎛 Filters")
    selected_zones  = st.multiselect("Zones",        options=sorted(df["zone"].dropna().unique()),         default=[])
    selected_causes = st.multiselect("Event Causes", options=sorted(df["event_cause"].dropna().unique()),  default=[])
    filtered = df.copy()
    if selected_zones:  filtered = filtered[filtered["zone"].isin(selected_zones)]
    if selected_causes: filtered = filtered[filtered["event_cause"].isin(selected_causes)]
    st.caption(f"Showing **{len(filtered):,}** events")
    st.markdown("---")
    st.caption("Built for Smart Traffic Challenge · ASTRAM Dataset")

# ── HEADER ────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,rgba(124,58,237,0.2),rgba(0,229,255,0.05));
     border:1px solid #1e2d4a;border-radius:12px;padding:20px 28px;margin-bottom:24px;
     display:flex;align-items:center;gap:16px;'>
  <span style='font-size:2.5rem'>🚦</span>
  <div>
    <h1 style='margin:0;font-size:1.6rem;font-weight:700;letter-spacing:-0.5px'>
      ASTRAM Event Congestion Intelligence Platform
    </h1>
    <p style='margin:4px 0 0;color:#64748b;font-size:0.85rem;font-family:monospace'>
      EVENT-DRIVEN CONGESTION · FORECAST · RESOURCE DEPLOYMENT · DIVERSION PLANNING
    </p>
  </div>
</div>""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "⏱ Temporal Patterns", "⚠️ Risk Matrix",
    "🤖 AI Event Planner", "📈 Impact Simulator", "🗺 Diversion Playbook",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW  +  FEATURE A: TODAY'S RISK INDEX
# ══════════════════════════════════════════════════════════════════
with tab1:

    # ── FEATURE A: TODAY'S RISK INDEX ──
    score, level, color, cur_hour, cur_dow = compute_today_risk(df)
    st.markdown("#### 🔴 Today's Live Congestion Risk Index")

    gauge_col, detail_col = st.columns([1, 2])
    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{level}</b><br><span style='font-size:0.75rem;color:#64748b'>{cur_dow} · {cur_hour:02d}:00</span>",
                   "font": {"color": color, "size": 16}},
            gauge={
                "axis": {"range":[0,100], "tickcolor":"#64748b"},
                "bar":  {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range":[0,35],  "color":"rgba(16,185,129,0.15)"},
                    {"range":[35,55], "color":"rgba(0,229,255,0.15)"},
                    {"range":[55,75], "color":"rgba(245,158,11,0.15)"},
                    {"range":[75,100],"color":"rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line":{"color":color,"width":3},"thickness":0.8,"value":score},
            },
            number={"font":{"color":color,"size":42}, "suffix":"/100"},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=240,
                                 margin=dict(l=20,r=20,t=20,b=10),
                                 font=dict(color="#94a3b8"))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with detail_col:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#1a2540);border:1px solid #1e2d4a;
             border-left:4px solid {color};border-radius:10px;padding:18px;margin-top:8px'>
          <div style='font-size:0.7rem;color:#64748b;letter-spacing:2px;font-family:monospace'>RISK BREAKDOWN</div>
          <div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:10px'>
            <div>
              <div style='font-size:0.7rem;color:#64748b'>HOUR FACTOR</div>
              <div style='font-size:1.3rem;font-weight:700;color:#00e5ff'>{cur_hour:02d}:00</div>
              <div style='font-size:0.75rem;color:#64748b'>{"Peak hour — very high risk" if cur_hour in range(20,23) else "Off-peak" if cur_hour in range(10,17) else "Elevated risk"}</div>
            </div>
            <div>
              <div style='font-size:0.7rem;color:#64748b'>DAY FACTOR</div>
              <div style='font-size:1.3rem;font-weight:700;color:#7c3aed'>{cur_dow}</div>
              <div style='font-size:0.75rem;color:#64748b'>{"Highest risk day" if cur_dow=="Friday" else "Weekend — lower risk" if cur_dow in ["Saturday","Sunday"] else "Weekday risk"}</div>
            </div>
            <div>
              <div style='font-size:0.7rem;color:#64748b'>RECOMMENDED ACTION</div>
              <div style='font-size:0.9rem;font-weight:600;color:{color}'>
                {"🚨 Pre-deploy all units" if level=="CRITICAL" else "⚠️ Standby units ready" if level=="HIGH" else "✅ Normal monitoring" if level=="LOW" else "👁 Elevated monitoring"}
              </div>
            </div>
            <div>
              <div style='font-size:0.7rem;color:#64748b'>COMPOSITE SCORE</div>
              <div style='font-size:1.3rem;font-weight:700;color:{color}'>{score}/100</div>
              <div style='font-size:0.75rem;color:#64748b'>Based on 8,173 historical events</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Key Metrics")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, val, label, sub, color in [
        (c1, f"{len(filtered):,}",                                  "Total Events",   "Nov 2023 – Apr 2024",                         "#00e5ff"),
        (c2, f"{filtered['event_type'].eq('unplanned').sum():,}",    "Unplanned",      f"{filtered['event_type'].eq('unplanned').mean()*100:.1f}% of total","#ef4444"),
        (c3, f"{filtered['event_type'].eq('planned').sum():,}",      "Planned",        "Higher closure risk",                         "#a78bfa"),
        (c4, f"{filtered['requires_road_closure'].mean()*100:.1f}%", "Road Closures",  f"{int(filtered['requires_road_closure'].sum())} events closed","#f59e0b"),
        (c5, f"{filtered['priority'].eq('High').mean()*100:.1f}%",   "High Priority",  f"{filtered['priority'].eq('High').sum():,} events","#ef4444"),
        (c6, f"{filtered['duration_min'].median():.0f}m",            "Median Duration","Per resolved event",                          "#10b981"),
    ]:
        col.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-val' style='color:{color}'>{val}</div>
          <div class='metric-sub'>{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Events by Cause")
        cc = filtered["event_cause"].value_counts().head(12).reset_index()
        cc.columns = ["Cause","Count"]
        cc["Cause"] = cc["Cause"].str.replace("_"," ").str.title()
        fig = px.bar(cc, x="Count", y="Cause", orientation="h",
                     color="Count", color_continuous_scale=["#1a2540","#7c3aed","#00e5ff"])
        fig.update_layout(**PT, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("#### Events by Zone")
        zc = filtered["zone"].value_counts().dropna().reset_index()
        zc.columns = ["Zone","Count"]
        fig = px.pie(zc, names="Zone", values="Count", color_discrete_sequence=PAL, hole=0.55)
        fig.update_layout(**PT, height=380)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Road Closure Rate by Event Cause (%)")
        cl = (filtered.groupby("event_cause")["requires_road_closure"]
              .mean().mul(100).round(1).sort_values(ascending=True).reset_index())
        cl.columns = ["Cause","Closure %"]
        cl["Cause"] = cl["Cause"].str.replace("_"," ").str.title()
        cl["Color"] = cl["Closure %"].apply(lambda v:"#ef4444" if v>50 else "#f59e0b" if v>25 else "#00e5ff")
        fig = go.Figure(go.Bar(x=cl["Closure %"], y=cl["Cause"], orientation="h",
                               marker_color=cl["Color"], marker_line_width=0))
        fig.update_layout(**PT, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        st.markdown("#### Top Corridors by Event Volume")
        corr = (filtered[filtered["corridor"]!="Non-corridor"]["corridor"]
                .value_counts().head(10).reset_index())
        corr.columns = ["Corridor","Count"]
        fig = px.bar(corr, x="Corridor", y="Count",
                     color="Count", color_continuous_scale=["#1a2540","#7c3aed","#a78bfa"])
        fig.update_layout(**PT, coloraxis_showscale=False, height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Hourly Event Distribution")
    hc = filtered["hour"].value_counts().sort_index().reset_index()
    hc.columns = ["Hour","Count"]
    hc["Label"] = hc["Hour"].apply(lambda h: f"{h:02d}:00")
    hc["Color"] = hc["Count"].apply(lambda v:"#ef4444" if v>700 else "#f59e0b" if v>500 else "#00e5ff")
    fig = go.Figure(go.Bar(x=hc["Label"], y=hc["Count"], marker_color=hc["Color"], marker_line_width=0))
    fig.update_layout(**PT, height=250, xaxis_title="Hour of Day", yaxis_title="Event Count")
    st.plotly_chart(fig, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### Events by Day of Week")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dc = filtered["day_of_week"].value_counts().reindex(day_order, fill_value=0).reset_index()
        dc.columns = ["Day","Count"]
        fig = px.bar(dc, x="Day", y="Count", color="Count",
                     color_continuous_scale=["#1a2540","#7c3aed","#00e5ff"])
        fig.update_layout(**PT, coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col_f:
        st.markdown("#### Median Resolution Time by Cause (log scale)")
        dur = (filtered.groupby("event_cause")["duration_min"]
               .median().dropna().sort_values(ascending=True).reset_index())
        dur.columns = ["Cause","Median Minutes"]
        dur["Cause"] = dur["Cause"].str.replace("_"," ").str.title()
        fig = px.bar(dur, x="Median Minutes", y="Cause", orientation="h", log_x=True,
                     color="Median Minutes", color_continuous_scale=["#1a2540","#ff6b35","#ef4444"])
        fig.update_layout(**PT, coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Zone × Hour Heatmap")
    zones_h = sorted(filtered["zone"].dropna().unique())
    z_mat = [[len(filtered[(filtered["zone"]==z) & (filtered["hour"]==h)]) for h in range(24)] for z in zones_h]
    fig = go.Figure(go.Heatmap(
        z=z_mat, x=[f"{h:02d}h" for h in range(24)], y=zones_h,
        colorscale=[[0,"#0a0f1e"],[0.3,"rgba(0,229,255,0.4)"],[0.7,"rgba(255,107,53,0.7)"],[1,"#ef4444"]],
        showscale=True,
        hovertemplate="Zone: %{y}<br>Hour: %{x}<br>Events: %{z}<extra></extra>",
    ))
    fig.update_layout(**PT, height=320, xaxis_title="Hour of Day")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — RISK MATRIX
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Event Risk Matrix")
    st.caption("Quantified scoring across three dimensions — replacing experience-driven guesswork with data-backed deployment thresholds.")
    risk_df = pd.DataFrame(RISK_DATA)
    risk_df["Severity Score"] = (
        risk_df["Closure %"]*0.4 + risk_df["Duration Score"]*0.35 + risk_df["Manpower Score"]*0.25
    ).div(10).round(1)
    display_cols = ["Cause","Closure %","Duration Score","Manpower Score","Severity Score","Police Units","Barricades","Priority"]
    st.dataframe(
        risk_df[display_cols], use_container_width=True, hide_index=True,
        column_config={
            "Closure %":      st.column_config.ProgressColumn("Closure Risk %", min_value=0, max_value=100, format="%d%%"),
            "Severity Score": st.column_config.ProgressColumn("Severity Score", min_value=0, max_value=10,  format="%.1f"),
        }, height=440,
    )
    st.markdown("#### Radar: Closure Risk vs Duration Impact vs Manpower Need")
    causes_r = [r["Cause"] for r in RISK_DATA[:10]]
    fig = go.Figure()
    for label, key, lc, fc in [
        ("Closure Risk",    "Closure %",      "#ef4444","rgba(239,68,68,0.12)"),
        ("Duration Impact", "Duration Score", "#f59e0b","rgba(245,158,11,0.12)"),
        ("Manpower Need",   "Manpower Score", "#00e5ff","rgba(0,229,255,0.12)"),
    ]:
        vals = [r[key] for r in RISK_DATA[:10]]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=causes_r+[causes_r[0]],
                                      name=label, line_color=lc, fill="toself", fillcolor=fc))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, color="#64748b", gridcolor="#1e2d4a"),
                   angularaxis=dict(color="#64748b", gridcolor="#1e2d4a")),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", family="Space Grotesk"),
        legend=dict(font=dict(color="#94a3b8")), height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — AI EVENT PLANNER  +  FEATURE B + D
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 AI-Powered Event Deployment Planner")
    st.caption("GPT-4o generates a deployment plan grounded in 8,173 historical incidents · Similar past events shown · PDF report available")

    SYSTEM_PROMPT = """You are an expert traffic management AI for ASTRAM (Bengaluru, India).

Historical data from 8,173 events (Nov 2023 – Apr 2024):
- Road closure rates: vip_movement 80%, public_event 46%, protest 40%, procession 26%, construction 27%
- Peak congestion hours: 20:00–22:00 (800+ events/hr), 4–7 AM (660+/hr)
- Top corridors: Mysore Road (743), Bellary Road 1 (610), Tumkur Road (458)
- Median resolution: 64.5 min; construction ~50 hrs; tree fall ~12 hrs

Respond with EXACTLY 4 sections using markdown headers:

## 📊 Impact Forecast
(closure probability %, congestion radius, duration estimate, peak window)

## 👮 Manpower Deployment Plan
(exact police units, traffic wardens, PCR vans, positions, timings)

## 🚧 Barricading Strategy
(barricade set count, placement points, advance warning distances, VMS activation)

## 🗺️ Diversion Routes
(primary + secondary alternate routes with estimated added travel time)

Be specific with numbers. Reference Bengaluru roads. Keep each section to 4-5 lines."""

    with st.form("planner_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            evt_type = st.selectbox("Event Type *", [
                "","Political Rally","Religious Procession","Sports Match","Music Festival",
                "VIP Movement","Protest / Bandh","Cultural Festival","Marathon / Run","Road Construction",
            ])
            evt_zone = st.selectbox("Zone *", [""] + sorted(df["zone"].dropna().unique().tolist()))
        with col2:
            evt_corridor = st.selectbox("Corridor", [""] + sorted(
                df[df["corridor"]!="Non-corridor"]["corridor"].dropna().unique().tolist()))
            evt_time = st.selectbox("Expected Time", [
                "","Early Morning (4–7 AM)","Morning Peak (7–10 AM)","Midday (10 AM–2 PM)",
                "Evening Peak (5–8 PM)","Night (8 PM–12 AM)","Late Night (12–4 AM)",
            ])
        with col3:
            crowd_size   = st.text_input("Expected Crowd Size", placeholder="e.g. 5000")
            duration_est = st.text_input("Duration (hours)", placeholder="e.g. 3")
        submitted = st.form_submit_button("⚡ Generate Deployment Plan", use_container_width=True)

    if submitted:
        if not evt_type or not evt_zone:
            st.error("Please select at least an Event Type and Zone.")
        elif not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not found in .env file.")
        else:
            user_msg = (f"Event Type: {evt_type}\nZone: {evt_zone}\n"
                        f"Corridor: {evt_corridor or 'Not specified'}\nTime: {evt_time or 'Not specified'}\n"
                        f"Crowd Size: {crowd_size or 'Not specified'}\nDuration: {duration_est or 'Not specified'} hours\n\n"
                        "Generate the complete operational traffic management plan.")

            # Run AI + similarity in parallel columns
            with st.spinner("Generating plan with GPT-4o + finding similar incidents..."):
                ai_result   = call_openai(user_msg, SYSTEM_PROMPT)
                sim_events, sim_scores = find_similar_events(df, evt_type, evt_zone, evt_corridor, evt_time)
                sim_data    = compute_impact_simulation(df, evt_type)

            if ai_result.startswith("❌"):
                st.error(ai_result)
            else:
                st.success("✅ Plan generated")
                st.markdown(f"""
                <div style='background:linear-gradient(90deg,rgba(124,58,237,0.15),rgba(0,229,255,0.05));
                     border:1px solid #1e2d4a;border-radius:10px;padding:14px 20px;margin-bottom:16px'>
                  <strong>📋 Event:</strong> {evt_type} &nbsp;·&nbsp;
                  <strong>Zone:</strong> {evt_zone} &nbsp;·&nbsp;
                  <strong>Corridor:</strong> {evt_corridor or 'N/A'} &nbsp;·&nbsp;
                  <strong>Time:</strong> {evt_time or 'N/A'} &nbsp;·&nbsp;
                  <strong>Crowd:</strong> {crowd_size or 'N/A'}
                </div>""", unsafe_allow_html=True)

                plan_col, similar_col = st.columns([3, 2])

                with plan_col:
                    st.markdown("#### 📋 Deployment Plan")
                    st.markdown(ai_result)

                with similar_col:
                    # ── FEATURE B: SIMILAR EVENTS ──
                    st.markdown("#### 🔍 Most Similar Past Incidents")
                    st.caption("Evidence base for this recommendation")
                    for i, (_, row) in enumerate(sim_events.iterrows()):
                        closed = "🔴 YES" if row.get("requires_road_closure") else "🟢 NO"
                        dur    = f"{row['duration_min']:.0f} min" if pd.notna(row.get("duration_min")) else "N/A"
                        sim_pct = int(sim_scores[i]*100)
                        st.markdown(f"""
                        <div class='similar-card'>
                          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                            <strong style='color:#10b981;font-size:0.85rem'>#{i+1} Match — {sim_pct}% similar</strong>
                            <span style='font-size:0.75rem;color:#64748b'>{row.get("priority","N/A")} priority</span>
                          </div>
                          <div style='font-size:0.8rem;color:#94a3b8'>
                            <b>Cause:</b> {str(row.get("event_cause","")).replace("_"," ").title()}<br>
                            <b>Zone:</b> {row.get("zone","N/A")}<br>
                            <b>Hour:</b> {int(row.get("hour",0)):02d}:00 &nbsp;·&nbsp; <b>Day:</b> {row.get("day_of_week","N/A")}<br>
                            <b>Road Closed:</b> {closed} &nbsp;·&nbsp; <b>Duration:</b> {dur}
                          </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # ── FEATURE D: PDF REPORT ──
                st.markdown("#### 📥 Download Official Report")
                pdf_bytes = generate_pdf(
                    evt_type, evt_zone, evt_corridor, evt_time,
                    crowd_size, ai_result, sim_events, sim_data
                )
                st.download_button(
                    label="📄 Download PDF Deployment Report",
                    data=pdf_bytes,
                    file_name=f"ASTRAM_Plan_{evt_type.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.caption("PDF includes event summary, AI plan, similar incidents table, and impact simulation.")

    else:
        st.info("👆 Fill in event details and click **Generate Deployment Plan**.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class='ai-block'><h4>📊 IMPACT FORECAST</h4>
            <p style='font-size:0.85rem;color:#94a3b8'>Closure probability, congestion radius, duration, peak window — calibrated against 8,173 historical incidents.</p></div>
            <div class='ai-block'><h4>👮 MANPOWER PLAN</h4>
            <p style='font-size:0.85rem;color:#94a3b8'>Exact police units, wardens, PCR vans with positions and pre/during/post timings.</p></div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='ai-block'><h4>🔍 SIMILAR PAST INCIDENTS</h4>
            <p style='font-size:0.85rem;color:#94a3b8'>Top 3 most similar historical events with outcomes — evidence grounding for the AI recommendation.</p></div>
            <div class='ai-block'><h4>📄 PDF REPORT</h4>
            <p style='font-size:0.85rem;color:#94a3b8'>One-click PDF with event summary, full plan, historical evidence table, and impact simulation for offline use.</p></div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — FEATURE C: BEFORE/AFTER IMPACT SIMULATOR
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📈 Before vs After ASTRAM — Impact Simulator")
    st.caption("Select an event type to see the projected improvement in road closure rate and resolution time when ASTRAM is deployed vs no system.")

    sim_evt = st.selectbox("Select Event Type to Simulate", [
        "Political Rally","Religious Procession","Sports Match","Music Festival",
        "VIP Movement","Protest / Bandh","Cultural Festival","Marathon / Run","Road Construction",
    ], key="sim_evt")

    s = compute_impact_simulation(df, sim_evt)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, sub, color in [
        (k1, f"{s['base_closure']}%",  "Closure Rate (No System)", f"From {s['n_events']} historical events", "#ef4444"),
        (k2, f"{s['proj_closure']}%",  "Closure Rate (With ASTRAM)", f"↓ {s['closure_pct']}% reduction",    "#10b981"),
        (k3, f"{s['base_duration']}m", "Avg Duration (No System)",  "Median resolution time",                "#f59e0b"),
        (k4, f"{s['proj_duration']}m", "Avg Duration (With ASTRAM)", f"↓ {s['duration_pct']}% faster",      "#10b981"),
    ]:
        col.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-val' style='color:{color}'>{val}</div>
          <div class='metric-sub'>{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        st.markdown("#### Road Closure Rate: Before vs After")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Without ASTRAM", "With ASTRAM"],
            y=[s["base_closure"], s["proj_closure"]],
            marker_color=["#ef4444","#10b981"],
            marker_line_width=0,
            text=[f"{s['base_closure']}%", f"{s['proj_closure']}%"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=14),
        ))
        fig.add_annotation(
            x=0.5, y=max(s["base_closure"], s["proj_closure"])*0.6,
            text=f"<b>-{s['closure_saved']}%<br>closure events</b>",
            showarrow=False, font=dict(color="#10b981", size=14),
            xref="x domain",
        )
        fig.update_layout(**PT, height=320, yaxis_title="Closure Rate (%)",
                          yaxis_range=[0, s["base_closure"]*1.35])
        st.plotly_chart(fig, use_container_width=True)

    with col_sim2:
        st.markdown("#### Average Resolution Time: Before vs After")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Without ASTRAM", "With ASTRAM"],
            y=[s["base_duration"], s["proj_duration"]],
            marker_color=["#f59e0b","#10b981"],
            marker_line_width=0,
            text=[f"{s['base_duration']} min", f"{s['proj_duration']} min"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=14),
        ))
        fig.add_annotation(
            x=0.5, y=max(s["base_duration"], s["proj_duration"])*0.6,
            text=f"<b>-{s['time_saved']} min<br>per event</b>",
            showarrow=False, font=dict(color="#10b981", size=14),
            xref="x domain",
        )
        fig.update_layout(**PT, height=320, yaxis_title="Resolution Time (min)",
                          yaxis_range=[0, s["base_duration"]*1.35])
        st.plotly_chart(fig, use_container_width=True)

    # All event types comparison
    st.markdown("#### All Event Types — Projected Improvement Summary")
    all_evts = ["Political Rally","Religious Procession","VIP Movement",
                "Protest / Bandh","Cultural Festival","Marathon / Run","Road Construction"]
    rows = []
    for e in all_evts:
        s2 = compute_impact_simulation(df, e)
        rows.append({
            "Event Type": e,
            "Baseline Closure %": s2["base_closure"],
            "With ASTRAM %": s2["proj_closure"],
            "Closure Reduction": f"-{s2['closure_pct']}%",
            "Time Saved (min)": s2["time_saved"],
            "Duration Reduction": f"-{s2['duration_pct']}%",
            "Historical Events": s2["n_events"],
        })
    summary_df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Without ASTRAM", x=summary_df["Event Type"],
                         y=summary_df["Baseline Closure %"], marker_color="#ef4444", marker_line_width=0))
    fig.add_trace(go.Bar(name="With ASTRAM",    x=summary_df["Event Type"],
                         y=summary_df["With ASTRAM %"],    marker_color="#10b981", marker_line_width=0))
    fig.update_layout(**PT, barmode="group", height=340,
                      yaxis_title="Road Closure Rate (%)",
                      xaxis_tickangle=-20,
                      legend=dict(font=dict(color="#94a3b8")))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — DIVERSION PLAYBOOK
# ══════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🗺 Standard Diversion Playbooks")
    st.caption("Data-derived SOPs built from historical resolution patterns — the post-event learning system the challenge calls for.")

    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown("#### Event-Driven Closures by Zone")
        evt_df = filtered[filtered["event_cause"].isin(["public_event","procession","protest","vip_movement"])]
        ez = evt_df["zone"].value_counts().dropna().reset_index()
        ez.columns = ["Zone","Count"]
        fig = px.bar(ez, x="Zone", y="Count", color="Count",
                     color_continuous_scale=["#1a2540","#7c3aed","#00e5ff"])
        fig.update_layout(**PT, coloraxis_showscale=False, height=300, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with col_h:
        st.markdown("#### Event Closures by Corridor")
        ec = (evt_df[evt_df["corridor"]!="Non-corridor"]["corridor"]
              .value_counts().head(10).reset_index())
        ec.columns = ["Corridor","Count"]
        fig = px.bar(ec, x="Count", y="Corridor", orientation="h",
                     color="Count", color_continuous_scale=["#1a2540","#ff6b35","#ef4444"])
        fig.update_layout(**PT, coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📖 Protocol Library")
    playbooks = [
        ("🏛","Political Rally / Public Event","🔴 HIGH","46%",
         "5 Police Units · 8 Barricade Sets · 3 Signal Overrides · VMS Activation",[
         ("T-2h: Pre-Event Deployment","Deploy 5 police units at venue perimeter. Set up 8 barricade sets on primary ingress roads. Activate signal override on 3 nearest junctions."),
         ("Crowd Arrival Management","Stagger entries via Mysore Road alternate and Magadi Road. One-way flows on flanking streets. PCR van at each gate."),
         ("Post-Event Dispersal","Release crowds in phased 15-min intervals by zone. Keep Hosur Road clear as primary egress. Extend signal timing +40% on parallel roads for 90 min."),
         ("Incident Response","If road closure required: alternate via Bannerghata Road. Est. resolution: 36–90 min. Log post-event for pattern library."),
        ]),
        ("🚔","VIP Movement","🔴 HIGH","80%",
         "8 Police Units · 12 Barricade Sets · Full Corridor Lockdown · VMS at 2km/1km/500m",[
         ("T-30min: Route Lockdown","Clear entire movement corridor. Escort units at 500m intervals. All cross-junctions red for convoy duration."),
         ("Parallel Diversion Active","Push all traffic to secondary corridor. VMS signs at 2km, 1km, 500m approach points."),
         ("Restoration","Typical convoy: 15–25 min. Restore normal flow junction-by-junction from tail to head of route."),
        ]),
        ("🌳","Tree Fall / Debris","🟡 MEDIUM","39%",
         "2 Police Units · 4 Barricade Sets · Crane/Crew Dispatch · VMS Alert",[
         ("Immediate Response","First responder assesses blockage. Full closure → deploy 4 barricade sets within 10 min of report."),
         ("BBMP Clearance Crew","Dispatch crane/cutting crew. Historical median clearance: 732 min — communicate delay via VMS and app."),
         ("Diversion Maintenance","Nearest parallel road (~400m). Maintain until debris cleared and road surface inspected."),
        ]),
        ("🏗","Road Construction","🟡 MEDIUM","27%",
         "3 Police Units · 10 Barricade Sets · 48h Pre-Notify · Daily Inspection",[
         ("Pre-Approved Schedule","Pre-notify 48h ahead. Deploy 3 barricade sets Day 1; maintain for planned duration (~50 hrs median)."),
         ("Advance Signage","Static diversion signs at 1km and 500m. App push notification for corridor users."),
         ("Daily Inspection","Check barricades and signage each morning. Coordinate with contractor on progress and expected reopening."),
        ]),
    ]
    for icon, title, severity, closure, resources, steps in playbooks:
        with st.expander(f"{icon} {title} — Closure Risk: {closure} · {severity}", expanded=False):
            st.markdown(f"**Resources Required:** `{resources}`")
            st.markdown("---")
            for i, (stitle, sdesc) in enumerate(steps, 1):
                st.markdown(f"""<div class='playbook-card'>
                  <strong style='color:#00e5ff'>Step {i}: {stitle}</strong><br>
                  <span style='font-size:0.88rem;color:#94a3b8'>{sdesc}</span>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Post-Event Learning Summary")
    st.dataframe(pd.DataFrame([
        {"Event Type":"Public Event",    "Avg Duration (min)":"49",     "Closure Rate":"46%","Most Affected Zone":"Central Zone 2","Best Diversion":"Mysore Road → Old Madras Road"},
        {"Event Type":"VIP Movement",    "Avg Duration (min)":"22",     "Closure Rate":"80%","Most Affected Zone":"All zones",       "Best Diversion":"Full alternate corridor"},
        {"Event Type":"Procession",      "Avg Duration (min)":"37",     "Closure Rate":"26%","Most Affected Zone":"West Zone 1",     "Best Diversion":"Magadi Road parallel"},
        {"Event Type":"Protest / Bandh", "Avg Duration (min)":"31",     "Closure Rate":"40%","Most Affected Zone":"Central Zone 1",  "Best Diversion":"CBD bypass via ORR"},
        {"Event Type":"Construction",    "Avg Duration (min)":"~2,964", "Closure Rate":"27%","Most Affected Zone":"Multiple corridors","Best Diversion":"Pre-notified static alternate"},
    ]), use_container_width=True, hide_index=True)
