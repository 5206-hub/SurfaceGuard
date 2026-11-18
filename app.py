"""SurfaceGuard — Premium Industrial Inspection Dashboard."""
import json
import os
from io import BytesIO
from pathlib import Path

# Ensure relative paths resolve correctly regardless of launch CWD
os.chdir(Path(__file__).parent)

import numpy as np
import streamlit as st
from PIL import Image

from surfaceguard.reporting import (
    MASK_COLORS,
    build_inspection,
    export_payload,
    inspection_html_report,
    probability_heatmap,
    regions_csv,
)
from surfaceguard.vision import NAMES, classical, combine, decode_image, prepare_gray

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SurfaceGuard · Industrial Inspection",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Comprehensive CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
footer { visibility: hidden; }
div[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

/* ── App background ── */
.stApp { background: #0d0f14 !important; }
section[data-testid="stMain"] { background: #0d0f14 !important; }
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1480px !important;
}

/* ── Reduce default vertical gap ── */
.stVerticalBlock { gap: 0.4rem !important; }
div[data-testid="column"] > div { gap: 0.4rem !important; }

/* ── Tab navigation ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px !important;
    background: #111420 !important;
    padding: 4px !important;
    border-radius: 10px !important;
    border: 1px solid #1c2035 !important;
    margin-bottom: 1.75rem !important;
}
.stTabs [data-baseweb="tab"] {
    height: 36px !important;
    padding: 0 22px !important;
    border-radius: 7px !important;
    color: #5e677a !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #b0b8d4 !important; }
.stTabs [aria-selected="true"] {
    color: #e8ebf4 !important;
    background: #1a2040 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; height: 0 !important; }

/* ── st.metric cards ── */
div[data-testid="stMetric"] {
    background: #111420 !important;
    border: 1px solid #1c2035 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.5rem !important;
}
div[data-testid="stMetricLabel"] > div {
    color: #5e677a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetricValue"] > div {
    color: #e8ebf4 !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}
div[data-testid="stMetricDelta"] { margin-top: 2px !important; }
div[data-testid="stMetricDelta"] > div { font-size: 0.78rem !important; font-weight: 600 !important; }

/* ── File uploader ── */
div[data-testid="stFileUploaderDropzone"] {
    background: #0e1119 !important;
    border: 2px dashed #252d47 !important;
    border-radius: 14px !important;
    padding: 52px 24px !important;
    text-align: center !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    background: #131a2e !important;
    border-color: #ff6b2b !important;
}
div[data-testid="stFileUploaderDropzone"] svg { fill: #ff6b2b !important; color: #ff6b2b !important; }
div[data-testid="stFileUploaderDropzone"] > div > p { color: #7b82a0 !important; font-size: 0.9rem !important; }
div[data-testid="stFileUploaderDropzone"] small { color: #3d4460 !important; }
div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
    color: #7b82a0 !important;
}

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"] {
    background: #ff6b2b !important;
    border: 1px solid #ff6b2b !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #e55a1f !important;
    border-color: #e55a1f !important;
    box-shadow: 0 4px 18px rgba(255,107,43,0.35) !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button {
    background: #131a2e !important;
    border: 1px solid #252d47 !important;
    color: #b0b8d4 !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
    border-color: #ff6b2b !important;
    color: #e8ebf4 !important;
    background: #1a2040 !important;
}

/* ── Segmented control ── */
div[data-testid="stSegmentedControl"] > label { display: none !important; }
div[data-baseweb="button-group"] {
    background: #111420 !important;
    border: 1px solid #1c2035 !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
div[data-baseweb="button-group"] > div > button {
    border-radius: 7px !important;
    color: #5e677a !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 20px !important;
    transition: all 0.15s !important;
}
div[data-baseweb="button-group"] > div > button:hover { color: #b0b8d4 !important; }
div[data-baseweb="button-group"] > div > button[aria-pressed="true"] {
    background: #1a2040 !important;
    color: #e8ebf4 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #1c2035 !important;
}
div[data-testid="stDataFrame"] th {
    background: #111420 !important;
    color: #5e677a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Image captions ── */
div[data-testid="stImage"] p {
    color: #5e677a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    text-align: center !important;
    margin-top: 6px !important;
}

/* ── Info/Success/Warning/Error ── */
div[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.85rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #090c12 !important;
    border-right: 1px solid #1c2035 !important;
}

/* ── Select box ── */
div[data-testid="stSelectbox"] > label {
    color: #5e677a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > label {
    color: #5e677a !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Divider ── */
hr { border-color: #1c2035 !important; margin: 1.25rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #252d47; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #ff6b2b; }

/* ── Selection ── */
::selection { background: rgba(255,107,43,0.25); color: #fff; }
</style>
""", unsafe_allow_html=True)

# ─── Colour palette helpers ──────────────────────────────────────────────────
_SEV_COLOR = {
    "None":     ("#22c55e", "rgba(34,197,94,0.12)"),
    "Minimal":  ("#22c55e", "rgba(34,197,94,0.12)"),
    "Low":      ("#58a6ff", "rgba(88,166,255,0.12)"),
    "Medium":   ("#f59e0b", "rgba(245,158,11,0.12)"),
    "High":     ("#ef4444", "rgba(239,68,68,0.12)"),
    "Critical": ("#dc2626", "rgba(220,38,38,0.15)"),
}

def _severity(pct: float) -> str:
    if pct == 0:    return "None"
    if pct < 1.0:   return "Minimal"
    if pct < 5.0:   return "Low"
    if pct < 15.0:  return "Medium"
    if pct < 30.0:  return "High"
    return "Critical"

def _sev_badge(level: str) -> str:
    c, bg = _SEV_COLOR.get(level, ("#8892b0", "rgba(136,146,176,0.12)"))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;'
        f'background:{bg};color:{c};border:1px solid {c}40;'
        f'padding:5px 14px;border-radius:999px;font-size:0.78rem;'
        f'font-weight:700;letter-spacing:0.07em">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{c};'
        f'box-shadow:0 0 6px {c}80;display:inline-block"></span>'
        f'{level.upper()}</span>'
    )

def _kpi(label: str, value: str, sub: str = "", left_accent: bool = False) -> str:
    accent = "border-left:3px solid #ff6b2b;" if left_accent else ""
    return (
        f'<div style="background:#111420;border:1px solid #1c2035;border-radius:12px;'
        f'padding:20px 22px;{accent}">'
        f'<div style="color:#5e677a;font-size:0.7rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">{label}</div>'
        f'<div style="color:#e8ebf4;font-size:1.75rem;font-weight:700;line-height:1">{value}</div>'
        f'{"" if not sub else f"<div style=\"color:#5e677a;font-size:0.78rem;margin-top:6px\">{sub}</div>"}'
        f'</div>'
    )

def _section(title: str, subtitle: str = "") -> str:
    sub_html = (
        f'<span style="color:#5e677a;font-size:0.82rem;font-weight:400;'
        f'margin-left:12px;letter-spacing:0">{subtitle}</span>'
        if subtitle else ""
    )
    return (
        f'<div style="margin:0.5rem 0 1.1rem">'
        f'<span style="color:#e8ebf4;font-size:0.95rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.1em">{title}</span>'
        f'{sub_html}</div>'
    )

def _divider() -> str:
    return '<div style="border-top:1px solid #1c2035;margin:1.25rem 0"></div>'

def _card_wrap(content: str, padding: str = "24px") -> str:
    return (
        f'<div style="background:#111420;border:1px solid #1c2035;'
        f'border-radius:14px;padding:{padding}">{content}</div>'
    )

# ─── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model checkpoint…")
def _load_model(path: str, _mtime: float):
    from surfaceguard.network import load_checkpoint
    return load_checkpoint(path)

@st.cache_data(show_spinner=False)
def _load_evaluation() -> dict:
    p = Path("comparison_perclass_results.json")
    if not p.is_file():
        p = Path("results.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

@st.cache_data(show_spinner=False)
def _load_selection(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

# --- resolve active checkpoint ---
_MODEL_OPTIONS: dict[str, tuple[Path, Path]] = {}
if Path("models/best_perclass.pt").is_file() and Path("models/best_perclass_selection.json").is_file():
    _MODEL_OPTIONS["New per-label model"] = (
        Path("models/best_perclass.pt"), Path("models/best_perclass_selection.json"),
    )
if Path("models/best.pt").is_file():
    _MODEL_OPTIONS["Saved baseline"] = (Path("models/best.pt"), Path("models/selection.json"))

with st.sidebar:
    st.markdown(
        '<div style="color:#5e677a;font-size:0.7rem;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">'
        'Checkpoint</div>',
        unsafe_allow_html=True,
    )
    chosen = st.selectbox(
        "Checkpoint",
        list(_MODEL_OPTIONS),
        label_visibility="collapsed",
    ) if _MODEL_OPTIONS else None
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#5e677a;font-size:0.7rem;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">'
        'Classical threshold</div>',
        unsafe_allow_html=True,
    )
    cutoff = st.slider("Classical threshold", 0.15, 0.90, 0.65, 0.01, label_visibility="collapsed")

ckpt_path, sel_path = _MODEL_OPTIONS[chosen] if chosen else (Path("models/best.pt"), Path("models/selection.json"))

loaded = None
if ckpt_path.is_file():
    try:
        loaded = _load_model(str(ckpt_path), ckpt_path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Cannot load checkpoint: {exc}")

model, meta = loaded if loaded else (None, {})
selection = _load_selection(str(sel_path)) if loaded else {}

if selection and selection.get("checkpoint_dataset_digest") != meta.get("dataset_digest"):
    st.error("Selection file does not match the loaded checkpoint. Remove models/selection.json.")
    st.stop()

fg_thresh = float(selection.get("cnn_foreground_threshold", meta.get("threshold", 0.5)))
hybrid_w  = float(selection.get("hybrid_cnn_weight",       meta.get("fusion_weight", 0.6)))

evaluation = _load_evaluation()

# ─── Navigation ─────────────────────────────────────────────────────────────
TAB_OVERVIEW, TAB_INSPECT, TAB_EVAL, TAB_ABOUT = st.tabs([
    "⬡   Overview",
    "🔍   Inspection",
    "📊   Model Evaluation",
    "ℹ   About",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with TAB_OVERVIEW:
    # Hero
    st.markdown(
        '<div style="margin-bottom:2rem">'
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:10px">'
        '<span style="font-size:2.2rem;line-height:1">⬡</span>'
        '<h1 style="margin:0;font-size:2.1rem;font-weight:800;color:#e8ebf4;'
        'letter-spacing:-0.03em;line-height:1">SurfaceGuard</h1>'
        '</div>'
        '<p style="color:#7b82a0;font-size:0.95rem;margin:0;max-width:580px;line-height:1.6">'
        'AI-powered surface defect detection and classification for hot-rolled steel strip '
        'inspection. Combines a classical OpenCV pipeline with a trained TinyUNet CNN for '
        'accurate pixel-level segmentation.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    best_dice = "—"
    if evaluation:
        try:
            nm = evaluation.get("new_model", evaluation)
            best_dice = f'{nm.get("hybrid_binary", nm.get("cnn_binary", {})).get("dice", 0):.4f}'
        except Exception:
            pass

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi("Models loaded", str(len(_MODEL_OPTIONS)), "checkpoints available"), unsafe_allow_html=True)
    k2.markdown(_kpi("Best Dice score", best_dice, "new hybrid · 840 test images", left_accent=True), unsafe_allow_html=True)
    k3.markdown(_kpi("Dataset size", "4,470", "paired image/mask samples"), unsafe_allow_html=True)
    k4.markdown(_kpi("Detection methods", "3", "Classical · CNN · Hybrid"), unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown(_section("Active Checkpoint"), unsafe_allow_html=True)
        if loaded:
            ep  = meta.get("epoch", "unknown")
            trn = meta.get("training_images", "?")
            vl  = meta.get("validation_images", selection.get("validation_images", "?"))
            epoch_badge = (
                f'<span style="background:#1a2040;color:#ff6b2b;border:1px solid #ff6b2b40;'
                f'padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;'
                f'letter-spacing:0.05em">Epoch {ep}</span>'
            )
            st.markdown(
                _card_wrap(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:flex-start;margin-bottom:14px">'
                    f'<div style="color:#e8ebf4;font-size:1rem;font-weight:700">{chosen}</div>'
                    f'{epoch_badge}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
                    f'<div style="background:#0d1018;border-radius:8px;padding:12px 14px">'
                    f'<div style="color:#5e677a;font-size:0.68rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">'
                    f'CNN Threshold</div>'
                    f'<div style="color:#e8ebf4;font-size:1.1rem;font-weight:700">{fg_thresh:.2f}</div>'
                    f'</div>'
                    f'<div style="background:#0d1018;border-radius:8px;padding:12px 14px">'
                    f'<div style="color:#5e677a;font-size:0.68rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">'
                    f'Hybrid CNN Weight</div>'
                    f'<div style="color:#e8ebf4;font-size:1.1rem;font-weight:700">{hybrid_w:.2f}</div>'
                    f'</div>'
                    f'<div style="background:#0d1018;border-radius:8px;padding:12px 14px">'
                    f'<div style="color:#5e677a;font-size:0.68rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">'
                    f'Training Images</div>'
                    f'<div style="color:#e8ebf4;font-size:1.1rem;font-weight:700">{trn:,}</div>'
                    f'</div>'
                    f'<div style="background:#0d1018;border-radius:8px;padding:12px 14px">'
                    f'<div style="color:#5e677a;font-size:0.68rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">'
                    f'Validation Images</div>'
                    f'<div style="color:#e8ebf4;font-size:1.1rem;font-weight:700">{vl:,}</div>'
                    f'</div>'
                    f'</div>'
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No checkpoint loaded. Classical baseline active.")

    with right_col:
        st.markdown(_section("Pipeline Architecture"), unsafe_allow_html=True)
        steps = [
            ("01", "Input",         "Grayscale 200×200 steel strip image"),
            ("02", "Classical",     "CLAHE enhancement + morphological filter"),
            ("03", "CNN",           "TinyUNet 4-class pixel segmentation"),
            ("04", "Fusion",        "Weighted score combination (hybrid mode)"),
            ("05", "Output",        "Labelled mask + region analysis + severity"),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:center;gap:14px;'
            f'padding:10px 0;border-bottom:1px solid #1c2035">'
            f'<span style="color:#ff6b2b;font-size:0.68rem;font-weight:800;'
            f'letter-spacing:0.08em;min-width:24px">{num}</span>'
            f'<div><div style="color:#e8ebf4;font-size:0.84rem;font-weight:600">{name}</div>'
            f'<div style="color:#5e677a;font-size:0.76rem;margin-top:1px">{desc}</div></div>'
            f'</div>'
            for num, name, desc in steps
        )
        st.markdown(_card_wrap(rows, "16px 20px"), unsafe_allow_html=True)

    if evaluation:
        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
        st.markdown(
            _section("Saved Test Metrics", "840 official held-out images · first 150 were inspected during earlier development"),
            unsafe_allow_html=True,
        )

        rows_html = ""
        def _perf_row(label, m, highlight=False):
            bg = "background:#1a2040;" if highlight else ""
            p = m.get("precision", 0) or 0
            r = m.get("recall", 0) or 0
            d = m.get("dice", 0) or 0
            iou = m.get("iou", 0) or 0
            bar_w = int(d * 100)
            return (
                f'<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1.5fr 1fr;'
                f'align-items:center;padding:10px 16px;border-bottom:1px solid #1c2035;'
                f'gap:8px;{bg}">'
                f'<div style="color:#{"e8ebf4" if highlight else "b0b8d4"};'
                f'font-size:0.84rem;font-weight:{"700" if highlight else "500"}">{label}</div>'
                f'<div style="color:#7b82a0;font-size:0.82rem;text-align:right">{p:.4f}</div>'
                f'<div style="color:#7b82a0;font-size:0.82rem;text-align:right">{r:.4f}</div>'
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="flex:1;height:4px;background:#1c2035;border-radius:2px;overflow:hidden">'
                f'<div style="width:{bar_w}%;height:100%;background:{"#ff6b2b" if highlight else "#3a4060"};'
                f'border-radius:2px"></div></div>'
                f'<span style="color:#{"ff6b2b" if highlight else "7b82a0"};'
                f'font-size:0.82rem;font-weight:{"700" if highlight else "500"};min-width:42px;text-align:right">'
                f'{d:.4f}</span></div>'
                f'<div style="color:#7b82a0;font-size:0.82rem;text-align:right">{iou:.4f}</div>'
                f'</div>'
            )

        header = (
            '<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1.5fr 1fr;'
            'padding:8px 16px;gap:8px;border-bottom:2px solid #252d47">'
            '<div style="color:#5e677a;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em">Method</div>'
            '<div style="color:#5e677a;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;text-align:right">Precision</div>'
            '<div style="color:#5e677a;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;text-align:right">Recall</div>'
            '<div style="color:#5e677a;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em">Dice</div>'
            '<div style="color:#5e677a;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;text-align:right">IoU</div>'
            '</div>'
        )
        rows_html += _perf_row("Classical", evaluation.get("classical_binary", {}))
        if "saved_baseline" in evaluation and "new_model" in evaluation:
            rows_html += _perf_row("Saved baseline · CNN", evaluation["saved_baseline"].get("cnn_binary", {}))
            rows_html += _perf_row("Saved baseline · Hybrid", evaluation["saved_baseline"].get("hybrid_binary", {}))
            rows_html += _perf_row("New model · CNN", evaluation["new_model"].get("cnn_binary", {}))
            rows_html += _perf_row("New model · Hybrid ★", evaluation["new_model"].get("hybrid_binary", {}), highlight=True)
        else:
            rows_html += _perf_row("CNN", evaluation.get("cnn_binary", {}))
            rows_html += _perf_row("Hybrid ★", evaluation.get("hybrid_binary", {}), highlight=True)

        st.markdown(
            f'<div style="background:#111420;border:1px solid #1c2035;border-radius:12px;overflow:hidden">'
            f'{header}{rows_html}</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════
with TAB_INSPECT:

    # ── Method selector ────────────────────────────────────────────────────
    st.markdown(_section("Detection Method"), unsafe_allow_html=True)
    method_opts = ["Classical", "CNN", "Hybrid"] if loaded else ["Classical"]
    method = st.segmented_control(
        "Method",
        method_opts,
        default=method_opts[-1],
        label_visibility="collapsed",
    )
    if method is None:
        method = method_opts[-1]

    # Method description cards
    _METHOD_INFO = {
        "Classical": ("OpenCV pipeline", "CLAHE contrast enhancement + morphological top-hat & black-hat filter. Fast, interpretable, no training data required. Outputs unclassified binary mask."),
        "CNN":       ("TinyUNet segmentation", "Compact U-Net with 4-class pixel-level output trained on NEU-SEG. Classifies each pixel as background or one of three defect label types."),
        "Hybrid":    ("CNN + Classical fusion", f"Weighted combination of CNN defect probability and classical score. CNN weight {hybrid_w:.0%}, classical weight {1-hybrid_w:.0%}. Improves robustness."),
    }
    if method in _METHOD_INFO:
        title, desc = _METHOD_INFO[method]
        badge = "★ Recommended" if method == "Hybrid" and loaded else ("Active" if not loaded else "")
        badge_html = (
            f'<span style="background:#ff6b2b18;color:#ff6b2b;border:1px solid #ff6b2b40;'
            f'padding:2px 10px;border-radius:6px;font-size:0.7rem;font-weight:700;'
            f'letter-spacing:0.05em;margin-left:10px">{badge}</span>'
            if badge else ""
        )
        st.markdown(
            f'<div style="background:#111420;border:1px solid #1c2035;border-left:3px solid #ff6b2b;'
            f'border-radius:10px;padding:12px 18px;margin-bottom:0.5rem">'
            f'<div style="display:flex;align-items:center;margin-bottom:6px">'
            f'<span style="color:#e8ebf4;font-size:0.85rem;font-weight:700">{title}</span>'
            f'{badge_html}</div>'
            f'<div style="color:#7b82a0;font-size:0.82rem;line-height:1.6">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # ── Upload + Demo area ─────────────────────────────────────────────────
    up_col, demo_col = st.columns([1.2, 1], gap="large")

    with up_col:
        st.markdown(_section("Upload Image"), unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload steel surface image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
        )

    with demo_col:
        st.markdown(_section("Example Images", "synthetic · for interface demo only"), unsafe_allow_html=True)
        _APP_DIR = Path(__file__).parent
        _DEMOS = {
            "Scratch":    _APP_DIR / "demo/demo_scratch.png",
            "Patch":      _APP_DIR / "demo/demo_patch.png",
            "Inclusion":  _APP_DIR / "demo/demo_inclusion.png",
        }
        avail = {k: v for k, v in _DEMOS.items() if v.is_file()}
        if not avail:
            st.markdown('<div style="color:#3d4460;font-size:0.82rem">No demo images found. Run <code>tools/make_demo.py</code>.</div>', unsafe_allow_html=True)
        dc = st.columns(max(len(avail), 1))
        for col, (name, path) in zip(dc, avail.items()):
            with col:
                try:
                    col.image(str(path), use_container_width=True, caption=name)
                except Exception:
                    pass
                if col.button(f"Inspect", key=f"demo_{name}", use_container_width=True):
                    st.session_state["_img_bytes"] = path.read_bytes()
                    st.session_state["_img_name"]  = f"{name} demo"
                    st.rerun()

    # ── Determine image to process ─────────────────────────────────────────
    img_bytes = None
    img_name  = None

    if uploaded_file is not None:
        new_id = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("_up_id") != new_id:
            st.session_state["_img_bytes"] = uploaded_file.getvalue()
            st.session_state["_img_name"]  = uploaded_file.name
            st.session_state["_up_id"]     = new_id
        img_bytes = st.session_state.get("_img_bytes")
        img_name  = st.session_state.get("_img_name", uploaded_file.name)
    elif "_img_bytes" in st.session_state:
        img_bytes = st.session_state["_img_bytes"]
        img_name  = st.session_state.get("_img_name", "image")

    _inspect_rgb = None
    if img_bytes is not None:
        try:
            _inspect_rgb = decode_image(img_bytes)
        except Exception as exc:
            st.error(f"Could not decode image: {exc}")

    if img_bytes is None or _inspect_rgb is None:
        if img_bytes is None:
            st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="text-align:center;padding:2rem 0;color:#3d4460;'
                'font-size:0.88rem">Upload an image or select a demo above to begin inspection.</div>',
                unsafe_allow_html=True,
            )
    else:
        rgb = _inspect_rgb
        try:
            gray = prepare_gray(rgb)
            with st.spinner("Running inspection pipeline…"):
                c_score, c_mask = classical(gray, threshold=cutoff)
                probabilities = None
                if method in ("CNN", "Hybrid") and model is not None:
                    from surfaceguard.network import predict as _predict
                    probabilities = _predict(model, gray)
                    cnn_weight = 1.0 if method == "CNN" else hybrid_w
                    labels = combine(probabilities, c_score, weight=cnn_weight, threshold=fg_thresh)
                else:
                    labels = c_mask
                class_names = NAMES if method != "Classical" else {**NAMES, 1: "Unclassified foreground"}
                inspection = build_inspection(
                    rgb,
                    labels,
                    probabilities,
                    img_name or "image",
                    method,
                    class_names=class_names,
                )
        except Exception as exc:
            st.error(f"Inspection failed: {exc}")
            st.stop()

        pct = inspection["affected_area_percent"]
        sev = _severity(pct)
        regions = inspection["regions"]
        n_reg = len(regions)
        labels = inspection["native_labels"]
        mask = inspection["coloured_mask"]
        view = inspection["annotated_overlay"]
        legacy_regions = []
        for region in regions:
            legacy = {
                "type": region["defect_class"],
                "x": region["bbox_x"],
                "y": region["bbox_y"],
                "width_px": region["bbox_width"],
                "height_px": region["bbox_height"],
                "area_px": region["defect_area_pixels"],
            }
            if "mean_class_probability" in region:
                legacy["confidence"] = region["mean_class_probability"]
            legacy_regions.append(legacy)
        report = {
            "affected_area_percent": pct,
            "affected_pixels": inspection["affected_area_pixels"],
            "inspected_pixels": inspection["inspected_pixels"],
            "regions": legacy_regions,
        }
        confidences = [region["mean_class_probability"] for region in regions if "mean_class_probability" in region]
        mean_conf = round(float(np.mean(confidences)), 3) if confidences else None

        # ── Inspection report header ───────────────────────────────────────────
        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
        mode_label = (
            f"{method} · Epoch {meta.get('epoch', '?')}" if method != "Classical" and loaded
            else method
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:1rem">'
            f'<div>'
            f'<span style="color:#e8ebf4;font-size:0.95rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em">Inspection Report · Model Prediction</span>'
            f'<span style="color:#5e677a;font-size:0.82rem;font-weight:400;margin-left:12px">'
            f'{img_name}</span>'
            f'</div>'
            f'<span style="background:#1a2040;color:#7b82a0;border:1px solid #252d47;'
            f'padding:4px 12px;border-radius:6px;font-size:0.76rem;font-weight:600">'
            f'{mode_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Three image panels ─────────────────────────────────────────────────
        ic1, ic2, ic3 = st.columns(3, gap="medium")
        with ic1:
            st.markdown(
                '<div style="background:#111420;border:1px solid #1c2035;border-radius:12px;overflow:hidden">',
                unsafe_allow_html=True,
            )
            st.image(rgb, use_container_width=True, caption="Original uploaded image · native resolution")
            st.markdown('</div>', unsafe_allow_html=True)
        with ic2:
            st.markdown(
                '<div style="background:#111420;border:1px solid #1c2035;border-radius:12px;overflow:hidden">',
                unsafe_allow_html=True,
            )
            st.image(mask, use_container_width=True, caption="MODEL PREDICTION · multiclass segmentation mask", clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with ic3:
            st.markdown(
                '<div style="background:#111420;border:1px solid #1c2035;border-radius:12px;overflow:hidden">',
                unsafe_allow_html=True,
            )
            st.image(view, use_container_width=True, caption="MODEL PREDICTION · native-resolution overlay and bounding boxes")
            st.markdown('</div>', unsafe_allow_html=True)

        legend_names = {0: "Background", **class_names}
        legend_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin:4px 14px 4px 0;color:#b0b8d4;font-size:0.8rem">'
            f'<i style="width:13px;height:13px;border-radius:3px;background:rgb{MASK_COLORS[label]};border:1px solid #5e677a;display:inline-block"></i>'
            f'{legend_names[label]}</span>'
            for label in (0, 1, 2, 3)
        )
        st.markdown(
            '<div style="background:#111420;border:1px solid #1c2035;border-radius:10px;padding:10px 14px;margin-top:10px">'
            '<span style="color:#5e677a;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-right:12px">'
            'Prediction legend</span>'
            + legend_html + '</div>',
            unsafe_allow_html=True,
        )
        inference_size = inspection["inference_resolution"]
        native_size = inspection["visualization_resolution"]
        st.caption(
            f"MODEL PREDICTION: inference ran at {inference_size['width']} x {inference_size['height']} px. "
            f"The class mask was resized with nearest-neighbour interpolation and the overlay, regions, and exports use the original "
            f"{native_size['width']} x {native_size['height']} px image resolution. This is not ground-truth annotation."
        )

        if inspection["native_probabilities"] is not None:
            with st.expander("Per-class softmax probability maps · MODEL PREDICTION", expanded=False):
                st.caption("Probability maps are linearly resized to original-image resolution and normalized per pixel.")
                probability_cols = st.columns(3)
                for col, label in zip(probability_cols, (1, 2, 3)):
                    col.image(
                        probability_heatmap(inspection["native_probabilities"][label], MASK_COLORS[label]),
                        caption=f"{NAMES[label]} probability",
                        use_container_width=True,
                    )

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        # ── Analysis + Region table ────────────────────────────────────────────
        analysis_col, table_col = st.columns([1, 1.6], gap="large")

        with analysis_col:
            st.markdown(_section("Defect Analysis"), unsafe_allow_html=True)

            # Severity block
            c, bg = _SEV_COLOR.get(sev, ("#8892b0", "rgba(136,146,176,0.1)"))
            st.markdown(
                f'<div style="background:{bg};border:1px solid {c}30;border-radius:12px;'
                f'padding:18px 20px;margin-bottom:12px;text-align:center">'
                f'<div style="color:#5e677a;font-size:0.68rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">Severity Heuristic</div>'
                f'<div style="display:flex;align-items:center;justify-content:center;gap:10px">'
                f'<div style="width:12px;height:12px;border-radius:50%;background:{c};'
                f'box-shadow:0 0 10px {c}80"></div>'
                f'<span style="color:{c};font-size:1.6rem;font-weight:800;letter-spacing:0.04em">'
                f'{sev.upper()}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption("Area-based display heuristic only; it is not an independently validated damage classification.")

            # Metric grid
            m1, m2 = st.columns(2)
            m1.metric("Affected Area",  f"{pct:.2f}%")
            m2.metric("Regions",        str(n_reg))
            if mean_conf is not None:
                m3, m4 = st.columns(2)
                m3.metric("Confidence",  f"{mean_conf:.3f}")
                detected_names = [NAMES[i] for i in (1, 2, 3) if (labels == i).any()]
                m4.metric("Labels",  str(len(detected_names)))

            if method != "Classical":
                detected = [NAMES[i] for i in (1, 2, 3) if (labels == i).any()]
                if detected:
                    label_html = " ".join(
                        f'<span style="background:#1a2040;color:#b0b8d4;border:1px solid #252d47;'
                        f'padding:3px 10px;border-radius:6px;font-size:0.76rem;font-weight:600">'
                        f'{n}</span>'
                        for n in detected
                    )
                    st.markdown(
                        f'<div style="margin-top:10px">'
                        f'<div style="color:#5e677a;font-size:0.68rem;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">Detected labels</div>'
                        f'{label_html}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="color:#5e677a;font-size:0.76rem;margin-top:10px;line-height:1.5">'
                    'Classical mode outputs unclassified binary foreground only. '
                    'Switch to CNN or Hybrid for defect label classification.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            import pandas as pd
            class_df = pd.DataFrame(inspection["class_summary"])[
                ["defect_class", "region_count", "area_pixels", "percent_of_original_image"]
            ].rename(
                columns={
                    "defect_class": "Predicted class",
                    "region_count": "Regions",
                    "area_pixels": "Area (px)",
                    "percent_of_original_image": "Image area (%)",
                }
            )
            st.markdown(_section("Class-wise prediction"), unsafe_allow_html=True)
            st.dataframe(class_df, hide_index=True, use_container_width=True)

            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

            # Downloads
            buf = BytesIO()
            Image.fromarray(view).save(buf, format="PNG")
            json_payload = export_payload(inspection, severity=sev)
            model_info = {
                "detection_method": method,
                "checkpoint": ckpt_path.name if loaded else "Classical pipeline only",
                "architecture": "TinyUNet, 4-class segmentation" if loaded else "OpenCV classical pipeline",
                "model_inference_resolution": f"{inference_size['width']} x {inference_size['height']} px",
                "native_visualisation_resolution": f"{native_size['width']} x {native_size['height']} px",
            }
            if loaded:
                model_info["checkpoint_epoch"] = meta.get("epoch", "unknown")
                model_info["cnn_foreground_threshold"] = f"{fg_thresh:.2f}"
            if method == "Hybrid":
                model_info["hybrid_cnn_weight"] = f"{hybrid_w:.2f}"
            html_report = inspection_html_report(inspection, model_info, severity=sev, class_names=class_names)
            dl1, dl2, dl3, dl4 = st.columns(4)
            dl1.download_button(
                "↓ PNG",
                data=buf.getvalue(),
                file_name="surfaceguard_prediction_overlay.png",
                mime="image/png",
                use_container_width=True,
            )
            dl2.download_button(
                "↓ JSON",
                data=json.dumps(json_payload, indent=2),
                file_name="surfaceguard_prediction.json",
                mime="application/json",
                use_container_width=True,
            )
            dl3.download_button(
                "CSV regions",
                data=regions_csv(inspection),
                file_name="surfaceguard_regions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            dl4.download_button(
                "HTML report",
                data=html_report,
                file_name="surfaceguard_inspection_report.html",
                mime="text/html",
                use_container_width=True,
            )

        with table_col:
            st.markdown(_section("Detected Regions", f"{n_reg} region{'s' if n_reg != 1 else ''} found"), unsafe_allow_html=True)
            if report["regions"]:
                import pandas as pd
                df = pd.DataFrame(report["regions"])
                # Rename for display
                rename = {
                    "type": "Type", "x": "X", "y": "Y",
                    "width_px": "W (px)", "height_px": "H (px)",
                    "area_px": "Area (px²)",
                }
                df = df.rename(columns=rename)
                col_cfg: dict = {
                    "Type":      st.column_config.TextColumn("Type",    width="medium"),
                    "X":         st.column_config.NumberColumn("X",      format="%d"),
                    "Y":         st.column_config.NumberColumn("Y",      format="%d"),
                    "W (px)":    st.column_config.NumberColumn("W px",   format="%d"),
                    "H (px)":    st.column_config.NumberColumn("H px",   format="%d"),
                    "Area (px²)":st.column_config.NumberColumn("Area px²", format="%d"),
                }
                if "confidence" in df.columns:
                    df = df.rename(columns={"confidence": "Confidence"})
                    col_cfg["Confidence"] = st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="%.3f",
                    )
                st.dataframe(df, column_config=col_cfg, hide_index=True, use_container_width=True)
                st.markdown(
                    '<div style="color:#3d4460;font-size:0.72rem;margin-top:6px">'
                    'Coordinates and areas use the original image resolution. '
                    'This table contains MODEL PREDICTION regions, not ground truth. '
                    + ('Confidence is the mean predicted class probability inside the region. ' if mean_conf is not None else '')
                    + '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:#111420;border:1px solid #1c2035;border-radius:12px;'
                    'padding:40px 24px;text-align:center;color:#3d4460;font-size:0.88rem">'
                    'No defect regions detected at the current threshold settings.'
                    '</div>',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with TAB_EVAL:
    st.markdown(
        '<div style="margin-bottom:1.5rem">'
        '<h2 style="color:#e8ebf4;font-size:1.3rem;font-weight:700;margin:0 0 6px">'
        'Model Evaluation</h2>'
        '<p style="color:#7b82a0;font-size:0.85rem;margin:0">'
        'Official held-out test set results. Settings were selected on the validation split only.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    if not evaluation:
        st.info("No evaluation results found. Run `compare_models.py` to generate them.")
    else:
        n_imgs = evaluation.get("images", "?")
        scope  = evaluation.get("evaluation_scope", "")
        prior  = (
            evaluation.get("test_set_history", {}).get("previous_150_image_subset_already_inspected")
            or evaluation.get("previous_test_subset", {}).get("already_inspected")
        )

        st.markdown(
            f'<div style="background:#111420;border:1px solid #1c2035;border-radius:10px;'
            f'padding:14px 18px;margin-bottom:1.25rem;display:flex;align-items:center;gap:14px">'
            f'<span style="font-size:1.3rem">📋</span>'
            f'<div><div style="color:#e8ebf4;font-size:0.85rem;font-weight:600">'
            f'Evaluated on {n_imgs} official test images</div>'
            f'<div style="color:#7b82a0;font-size:0.78rem;margin-top:3px">{scope}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if prior:
            st.warning(
                "⚠ The first 150 test images were inspected in an earlier project run. "
                "This 840-image report includes them and is not a fully fresh untouched estimate.",
                icon=None,
            )

        # Binary performance
        st.markdown(_section("Binary Detection Performance"), unsafe_allow_html=True)
        binary_rows = []
        def _add(label, m):
            binary_rows.append({
                "Method": label,
                "Precision": m.get("precision") or 0,
                "Recall": m.get("recall") or 0,
                "Dice ↑": m.get("dice") or 0,
                "IoU": m.get("iou") or 0,
                "GT pixels": m.get("ground_truth_pixels", ""),
                "Pred pixels": m.get("predicted_pixels", ""),
            })

        _add("Classical", evaluation.get("classical_binary", {}))
        if "saved_baseline" in evaluation and "new_model" in evaluation:
            for key, label in (("saved_baseline", "Saved baseline"), ("new_model", "New model")):
                _add(f"{label} · CNN",    evaluation[key].get("cnn_binary", {}))
                _add(f"{label} · Hybrid", evaluation[key].get("hybrid_binary", {}))
        else:
            _add("CNN",    evaluation.get("cnn_binary", {}))
            _add("Hybrid", evaluation.get("hybrid_binary", {}))

        import pandas as pd
        st.dataframe(
            pd.DataFrame(binary_rows),
            column_config={
                "Method":      st.column_config.TextColumn("Method", width="large"),
                "Precision":   st.column_config.NumberColumn("Precision", format="%.4f"),
                "Recall":      st.column_config.NumberColumn("Recall",    format="%.4f"),
                "Dice ↑":      st.column_config.ProgressColumn("Dice ↑", min_value=0, max_value=1, format="%.4f"),
                "IoU":         st.column_config.NumberColumn("IoU",       format="%.4f"),
                "GT pixels":   st.column_config.NumberColumn("GT pixels", format="%d"),
                "Pred pixels": st.column_config.NumberColumn("Pred pixels", format="%d"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Per-label
        label_rows = []
        sources = (
            [("Saved baseline", evaluation["saved_baseline"]), ("New model", evaluation["new_model"])]
            if "saved_baseline" in evaluation
            else [("Current model", evaluation)]
        )
        for src_name, src in sources:
            for meth_key, meth_label in (("cnn_per_label", "CNN"), ("hybrid_per_label", "Hybrid")):
                for lbl, m in src.get(meth_key, {}).items():
                    if isinstance(m, dict):
                        label_rows.append({
                            "Model": src_name,
                            "Method": meth_label,
                            "Label": lbl.replace("_", " ").title(),
                            "Precision": m.get("precision"),
                            "Recall": m.get("recall"),
                            "Dice ↑": m.get("dice"),
                            "IoU": m.get("iou"),
                        })

        if label_rows:
            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            st.markdown(_section("Per-Label Performance"), unsafe_allow_html=True)
            st.markdown(
                '<div style="color:#5e677a;font-size:0.78rem;margin-bottom:10px">'
                'Classical output is unclassified binary foreground; per-label scores apply to CNN and Hybrid only.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(label_rows),
                column_config={
                    "Model":     st.column_config.TextColumn("Model",  width="medium"),
                    "Method":    st.column_config.TextColumn("Method", width="small"),
                    "Label":     st.column_config.TextColumn("Label",  width="medium"),
                    "Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                    "Recall":    st.column_config.NumberColumn("Recall",    format="%.4f"),
                    "Dice ↑":    st.column_config.ProgressColumn("Dice ↑", min_value=0, max_value=1, format="%.4f"),
                    "IoU":       st.column_config.NumberColumn("IoU",       format="%.4f"),
                },
                hide_index=True,
                use_container_width=True,
            )

        # Model checkpoint cards
        if _MODEL_OPTIONS:
            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            st.markdown(_section("Checkpoint Details"), unsafe_allow_html=True)
            ccp = st.columns(len(_MODEL_OPTIONS))
            for col, (name, (cp, sp)) in zip(ccp, _MODEL_OPTIONS.items()):
                try:
                    m2, meta2 = _load_model(str(cp), cp.stat().st_mtime)
                    sel2 = _load_selection(str(sp))
                    ep2 = meta2.get("epoch", "unknown")
                    t2  = meta2.get("training_images", "?")
                    v2  = meta2.get("validation_images", sel2.get("validation_images", "?"))
                    thr2 = sel2.get("cnn_foreground_threshold", meta2.get("threshold", "?"))
                    hw2  = sel2.get("hybrid_cnn_weight", meta2.get("fusion_weight", "?"))
                    col.markdown(
                        _card_wrap(
                            f'<div style="color:#e8ebf4;font-size:0.88rem;font-weight:700;margin-bottom:12px">{name}</div>'
                            f'<div style="display:flex;flex-direction:column;gap:7px">'
                            + "".join(
                                f'<div style="display:flex;justify-content:space-between">'
                                f'<span style="color:#5e677a;font-size:0.76rem">{k}</span>'
                                f'<span style="color:#b0b8d4;font-size:0.76rem;font-weight:600">{v}</span>'
                                f'</div>'
                                for k, v in [
                                    ("Epoch", ep2), ("Train images", f"{t2:,}" if isinstance(t2, int) else t2),
                                    ("Val images",  f"{v2:,}" if isinstance(v2, int) else v2),
                                    ("CNN threshold", f"{float(thr2):.2f}" if thr2 != "?" else "?"),
                                    ("Hybrid CNN weight", f"{float(hw2):.2f}" if hw2 != "?" else "?"),
                                ]
                            )
                            + f'</div>',
                        ),
                        unsafe_allow_html=True,
                    )
                except Exception:
                    col.markdown(_kpi(name, "Error"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with TAB_ABOUT:
    left_a, right_a = st.columns(2, gap="large")

    with left_a:
        st.markdown(_section("Dataset"), unsafe_allow_html=True)
        st.markdown(
            _card_wrap(
                '<div style="color:#e8ebf4;font-size:0.95rem;font-weight:700;margin-bottom:14px">'
                'NEU-SEG Surface Defect Segmentation</div>'
                '<div style="display:flex;flex-direction:column;gap:9px">'
                + "".join(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'border-bottom:1px solid #1c2035;padding-bottom:9px">'
                    f'<span style="color:#5e677a;font-size:0.8rem">{k}</span>'
                    f'<span style="color:#b0b8d4;font-size:0.8rem;font-weight:600">{v}</span>'
                    f'</div>'
                    for k, v in [
                        ("Total samples",        "4,470 image/mask pairs"),
                        ("Image resolution",     "200 × 200 pixels"),
                        ("Image format",         "JPEG (grayscale)"),
                        ("Mask format",          "PNG (class IDs 0–3)"),
                        ("Training split",       "2,904 images"),
                        ("Validation split",     "726 images"),
                        ("Test split (held-out)","840 images"),
                        ("Mask labels",          "0 Background · 1,2,3 Defect types"),
                        ("Semantic names",       "Not published by dataset authors"),
                    ]
                )
                + '</div>',
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.markdown(_section("Limitations"), unsafe_allow_html=True)
        for lim in [
            "Affected area is a pixel-fraction estimate, not physical damage severity.",
            "NEU-SEG does not establish false-alarm rates on defect-free surfaces.",
            "Semantic defect names (scratch / patch / inclusion) are not assigned — no verified label mapping.",
            "Video processing is outside the scope of this prototype.",
            "The first 150 test images were inspected during earlier development runs.",
        ]:
            st.markdown(
                f'<div style="display:flex;gap:10px;padding:7px 0;'
                f'border-bottom:1px solid #111420">'
                f'<span style="color:#ff6b2b;margin-top:1px;font-size:0.8rem">—</span>'
                f'<span style="color:#7b82a0;font-size:0.82rem;line-height:1.5">{lim}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right_a:
        st.markdown(_section("Model Architecture"), unsafe_allow_html=True)
        st.markdown(
            _card_wrap(
                '<div style="color:#e8ebf4;font-size:0.95rem;font-weight:700;margin-bottom:14px">'
                'TinyUNet (4-class segmentation)</div>'
                '<div style="display:flex;flex-direction:column;gap:9px">'
                + "".join(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'border-bottom:1px solid #1c2035;padding-bottom:9px">'
                    f'<span style="color:#5e677a;font-size:0.8rem">{k}</span>'
                    f'<span style="color:#b0b8d4;font-size:0.8rem;font-weight:600">{v}</span>'
                    f'</div>'
                    for k, v in [
                        ("Architecture",   "Compact U-Net encoder–decoder"),
                        ("Encoder depth",  "3 blocks (width 12/24/48)"),
                        ("Output classes", "4 (background + labels 1,2,3)"),
                        ("Input",          "Single-channel 200×200 grayscale"),
                        ("Loss",           "Cross-entropy + soft Dice"),
                        ("Optimiser",      "Adam, lr = 0.001"),
                        ("Class weights",  "0.35 background · 1.4 × defect labels"),
                        ("Best epoch",     "3 (mean per-label Dice 0.7382)"),
                        ("Hardware",       "CPU (CUDA supported if available)"),
                        ("Parameters",     "66,976 trainable weights"),
                    ]
                )
                + '</div>',
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.markdown(_section("Technology Stack"), unsafe_allow_html=True)
        stack = [
            ("Python 3.13",          "Runtime"),
            ("PyTorch 2.15",         "Neural network inference & training"),
            ("OpenCV 4.14",          "Classical image processing pipeline"),
            ("Streamlit 1.67",       "Web application framework"),
            ("NumPy / Pillow",       "Array processing & image I/O"),
        ]
        st.markdown(
            _card_wrap(
                "".join(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:9px 0;border-bottom:1px solid #1c2035">'
                    f'<span style="color:#e8ebf4;font-size:0.82rem;font-weight:600">{lib}</span>'
                    f'<span style="color:#5e677a;font-size:0.76rem">{role}</span>'
                    f'</div>'
                    for lib, role in stack
                )
            ),
            unsafe_allow_html=True,
        )
