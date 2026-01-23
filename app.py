
import io
import json
import math
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

APP_NAME = "YJ - RCG Sales Reports"
SETTINGS_FILE = "yk_rcg_settings.json"

# RCG palette
RCG_PURPLE = "#2a206f"
RCG_BLUE = "#6498be"
RCG_BG = "#f7f7fb"
RCG_TEXT = "#111827"
RCG_MUTED = "#6b7280"
RCG_BORDER = "#e5e7eb"
RCG_CARD = "#ffffff"

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

# ---------- Branding / theme ----------
st.markdown(
    """
<style>
.stApp { background: #f7f7fb; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #2a206f;
  border-right: 0;
}
section[data-testid="stSidebar"] * { color: white !important; }

/* Sidebar widgets: white fields + dark text */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background: white !important;
  color: #111827 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: white !important;
  color: #111827 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #111827 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  background: white !important;
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.12);
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  background: #6498be !important;
  color: white !important;
  border: 0 !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
}

/* Sticky logo inside sidebar */
.rcg-sidebar-logo {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #2a206f;
  padding: 14px 12px 10px 12px;
  margin: -1rem -1rem 0 -1rem;
  border-bottom: 1px solid rgba(255,255,255,0.12);
}
.rcg-sidebar-appname{
  color: #ffffff;
  font-weight: 800;
  font-size: 0.95rem;
  text-align: center;
  margin-top: 6px;
  letter-spacing: 0.2px;
}
section[data-testid="stSidebar"] .stFileUploader, 
section[data-testid="stSidebar"] .stRadio, 
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stToggle{
  color: #ffffff !important;
}
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div{
  color: #ffffff;
}

/* Make multiselect tags match brand (no red) */
section[data-testid="stSidebar"] [data-baseweb="tag"]{
  background: #2a206f !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] svg{
  fill: #ffffff !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] [role="button"]{
  color: #ffffff !important;
}

/* Buttons in sidebar high contrast */
section[data-testid="stSidebar"] .stButton>button{
  background: #6498be !important;
  color: #ffffff !important;
  border: 0 !important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  filter: brightness(0.95);
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.35);
  background: white;
  color: #2a206f;
  font-weight: 800;
}

/* Sidebar buttons (SAVE/CLEAR etc) */
section[data-testid="stSidebar"] .stButton>button {
  background: #6498be !important;
  color: #ffffff !important;
  border: 0 !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
  filter: brightness(0.95);
}

/* Metric cards */
[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #6b7280; }
[data-testid="stMetricValue"] { color: #2a206f; }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-weight: 800; }
.stTabs [aria-selected="true"] { color: #2a206f !important; }

h1, h2, h3, h4 { color: #2a206f; }

/* Ensure text contrast on inputs in main area */
div[data-baseweb="select"] span { color: #111827 !important; }
input { color: #111827 !important; }

</style>
""",
    unsafe_allow_html=True,
)

pio.templates.default = "plotly_white"

# ---------- Settings persistence ----------
def load_settings() -> Dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings(settings: Dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

SETTINGS = load_settings()
from datetime import date, timedelta

def _last_full_month_range(today: date | None = None) -> Tuple[date, date]:
    """Return (first_day, last_day) of the last completed calendar month."""
    if today is None:
        today = date.today()
    first_this_month = today.replace(day=1)
    last_prev_month = first_this_month - timedelta(days=1)
    first_prev_month = last_prev_month.replace(day=1)
    return first_prev_month, last_prev_month


# ---------- Column expectations ----------
REQUIRED_COLUMNS = [
    "Sale site","Customer Name","Sale Reference","InvoiceReference","Product Name","Product Code",
    "Product Department","Quantity","Price excluding VAT","VAT","Discount","Restocking Fee","Total",
    "Total Without Vat","Sale Date","Invoice Date","Paid Amount","Buy Price","Channel","Employee",
]
NUMERIC_COLUMNS = [
    "Quantity","Price excluding VAT","VAT","Discount","Restocking Fee","Total","Total Without Vat","Paid Amount","Buy Price",
]
DATE_COLUMNS = ["Sale Date","Invoice Date"]
COLUMN_ALIASES = {"Invoice Reference": "InvoiceReference", "RestockingFee": "Restocking Fee"}

def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for src, target in COLUMN_ALIASES.items():
        if src in df.columns and target not in df.columns:
            rename_map[src] = target
    return df.rename(columns=rename_map) if rename_map else df

# ---------- Helpers ----------
def fmt_gbp(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"£{x:,.2f}"

def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{int(round(float(x))):,}"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _parse_dates_auto(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            s = df[c]
            if np.issubdtype(s.dtype, np.datetime64):
                continue
            d1 = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
            if d1.notna().mean() < 0.6:
                d2 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
                df[c] = d2 if d2.notna().mean() > d1.notna().mean() else d1
            else:
                df[c] = d1
    return df

def _ensure_required(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return df, missing

def _line_profit(row) -> float:
    q = row.get("Quantity", np.nan)
    sell = row.get("Price excluding VAT", np.nan)
    buy = row.get("Buy Price", np.nan)
    if pd.isna(q) or pd.isna(sell) or pd.isna(buy):
        return np.nan
    return (sell - buy) * q

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        for enc in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(file_bytes), engine="python")
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def build_base_df(file_bytes: bytes, filename: str, ignore_profit_everywhere: bool) -> Tuple[Optional[pd.DataFrame], List[str]]:
    raw = load_data(file_bytes, filename)
    raw = _normalize_columns(raw)
    raw = apply_column_aliases(raw)
    raw, missing = _ensure_required(raw)
    if missing:
        return None, missing

    df = _parse_dates_auto(_coerce_numeric(raw.copy(), NUMERIC_COLUMNS), DATE_COLUMNS)

    for c in ["Sale site","Employee","Channel","Product Department","Product Name","Product Code","Customer Name"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    df["Is Return"] = (
        (df["Quantity"].fillna(0) < 0)
        | (df["Total"].fillna(0) < 0)
        | (df["Total Without Vat"].fillna(0) < 0)
    )

    if not ignore_profit_everywhere:
        df["Line Profit"] = df.apply(_line_profit, axis=1)
    else:
        df["Line Profit"] = np.nan

    # Speed up groupby
    for c in ["Sale site","Employee","Channel","Product Department","Product Name","Customer Name"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df, []

def style_table(df: pd.DataFrame, money_cols: List[str], int_cols: List[str]):
    fmt_map = {}
    for c in money_cols:
        if c in df.columns:
            fmt_map[c] = lambda v, _c=c: fmt_gbp(v)
    for c in int_cols:
        if c in df.columns:
            fmt_map[c] = lambda v, _c=c: fmt_int(v)
    return df.style.format(fmt_map, na_rep="—")

def choose_chart_money(df_in: pd.DataFrame, chart_type: str, x: str, y: str, title: str, y_is_money: bool=True):
    if df_in.empty:
        st.info("No data to chart.")
        return
    if chart_type == "Line":
        fig = px.line(df_in, x=x, y=y, markers=True, title=title)
    elif chart_type == "Bar":
        fig = px.bar(df_in, x=x, y=y, title=title)
    elif chart_type == "Horizontal Bar":
        fig = px.bar(df_in, x=y, y=x, orientation="h", title=title)
    elif chart_type == "Pie":
        fig = px.pie(df_in, names=x, values=y, title=title)
    else:
        fig = px.bar(df_in, x=x, y=y, title=title)

    fig.update_layout(
        title_font_color=RCG_PURPLE,
        font_color=RCG_TEXT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    if y_is_money:
        if chart_type != "Pie":
            fig.update_yaxes(tickprefix="£", separatethousands=True)
            fig.update_traces(hovertemplate="%{x}<br>£%{y:,.2f}<extra></extra>")
        if chart_type == "Horizontal Bar":
            fig.update_traces(hovertemplate="%{y}<br>£%{x:,.2f}<extra></extra>")
        if chart_type == "Pie":
            fig.update_traces(hovertemplate="%{label}<br>£%{value:,.2f}<extra></extra>")
    else:
        fig.update_traces(hovertemplate="%{x}<br>%{y:,}<extra></extra>")
        if chart_type == "Horizontal Bar":
            fig.update_traces(hovertemplate="%{y}<br>%{x:,}<extra></extra>")
        if chart_type == "Pie":
            fig.update_traces(hovertemplate="%{label}<br>%{value:,}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

def product_rollup(df_in: pd.DataFrame) -> pd.DataFrame:
    g = df_in.groupby("Product Code", dropna=False, observed=True).agg(
        Product_Name=("Product Name", lambda s: s.mode().iloc[0] if not s.mode().empty else (s.dropna().astype(str).head(1).iloc[0] if len(s.dropna()) else "")),
        Department=("Product Department", lambda s: s.mode().iloc[0] if not s.mode().empty else (s.dropna().astype(str).head(1).iloc[0] if len(s.dropna()) else "")),
        Sales_Lines=("Product Code", "count"),
        Net_Units=("Quantity", "sum"),
        Total_Inc_VAT=("Total", "sum"),
        Total_Ex_VAT=("Total Without Vat", "sum"),
        VAT=("VAT", "sum"),
        Discount=("Discount", "sum"),
        Return_Lines=("Is Return", "sum"),
    ).reset_index()
    g["Product"] = g["Product_Name"].astype(str)
    return g

def summarize_group(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df_in.groupby(group_col, dropna=False, observed=True).agg(
        Sales_Lines=("Sale Reference", "count"),
        Net_Units=("Quantity", "sum"),
        Total_Inc_VAT=("Total", "sum"),
        Total_Ex_VAT=("Total Without Vat", "sum"),
        VAT=("VAT", "sum"),
        Discount=("Discount", "sum"),
        Return_Lines=("Is Return", "sum"),
    ).reset_index()
    return g

def invoice_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    inv = df_in.groupby("InvoiceReference", dropna=False, observed=True).agg(
        Sales_Lines=("Sale Reference","count"),
        Net_Units=("Quantity","sum"),
        Total_Inc_VAT=("Total","sum"),
        Total_Ex_VAT=("Total Without Vat","sum"),
    ).reset_index()
    return inv

def selection_block(kind: str, options: List[str], key: str) -> Optional[str]:
    st.caption(f"Quick select {kind}")
    q = st.text_input(f"Search {kind}", value=st.session_state.get(f"q_{key}", ""), key=f"q_{key}")
    ql = (q or "").strip().lower()
    opts = options
    if ql:
        opts = [o for o in options if ql in str(o).lower()][:200]
    sel = st.selectbox(f"Select {kind}", options=[""] + opts, index=0, key=f"sel_{key}")
    return sel if sel else None

def details_context_prompt() -> bool:
    pref = SETTINGS.get("details_apply_filters_pref", None)
    if pref in ("apply", "reset"):
        return pref == "apply"

    st.info("Apply current filters to this detail view, or reset to ALL data?")
    choice = st.radio("Detail view scope", ["Apply current filters", "Reset to ALL (ignore current filters)"], index=0)
    remember = st.checkbox("Remember my choice", value=False)
    if remember:
        SETTINGS["details_apply_filters_pref"] = "apply" if choice.startswith("Apply") else "reset"
        save_settings(SETTINGS)
    return choice.startswith("Apply")

def remember_recent(kind: str, value: str):
    if not value:
        return
    rec = SETTINGS.get("recent", {})
    arr = rec.get(kind, [])
    arr = [v for v in arr if v != value]
    arr.insert(0, value)
    rec[kind] = arr[:5]
    SETTINGS["recent"] = rec
    save_settings(SETTINGS)

def make_excel_report(df_filtered: pd.DataFrame, kpis_dict: Dict[str, str], tables_dict: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([{"Metric": k, "Value": v} for k, v in kpis_dict.items()]).to_excel(writer, index=False, sheet_name="Summary")
        for name, table in tables_dict.items():
            sn = name[:31]
            table.to_excel(writer, index=False, sheet_name=sn)
        df_filtered.to_excel(writer, index=False, sheet_name="Filtered Data")
        for sn in writer.sheets:
            writer.sheets[sn].freeze_panes(1, 0)
    output.seek(0)
    return output.getvalue()

# ---------- Header ----------
hl, hr = st.columns([1, 7], vertical_alignment="center")
with hl:
    try:
        st.image("logo.png", width=120)
    except Exception:
        pass
with hr:
    st.title(APP_NAME)
    st.caption("Fast • Filter • Drill-down • Exclusions • Exports • Returns aware")

# ---------- Sidebar ----------
with st.sidebar:
    # Fixed / sticky logo at top of sidebar (always above upload)
    st.markdown('<div class="rcg-sidebar-logo">', unsafe_allow_html=True)
    try:
        st.image("logo.png", width=180)
    except Exception:
        pass
    st.markdown('<div class="rcg-sidebar-appname">YJ - RCG Sales Reports</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.header("UPLOAD")
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.header("SETTINGS")
    revenue_basis = st.radio("Revenue metric", ["Total (incl VAT)", "Total Without Vat"], index=0)
    ignore_profit_everywhere = st.toggle("Ignore profit everywhere", value=True)
    trend_chart = st.selectbox("Trend chart type", ["Line", "Bar"], index=0)
    top_chart = st.selectbox("Top chart type", ["Horizontal Bar", "Bar", "Pie"], index=0)

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
df, missing = build_base_df(file_bytes, uploaded.name, ignore_profit_everywhere)
if missing:
    st.error("Missing required columns:\n\n" + "\n".join([f"- {c}" for c in missing]))
    st.stop()
assert df is not None

# Lists
all_customers = sorted([c for c in df["Customer Name"].astype(str).unique().tolist() if str(c).strip()])
all_employees = sorted([c for c in df["Employee"].astype(str).unique().tolist() if str(c).strip()])
all_branches = sorted([c for c in df["Sale site"].astype(str).unique().tolist() if str(c).strip()])
all_channels = sorted([c for c in df["Channel"].astype(str).unique().tolist() if str(c).strip()])
all_depts = sorted([c for c in df["Product Department"].astype(str).unique().tolist() if str(c).strip()])
all_products = sorted([c for c in df["Product Name"].astype(str).unique().tolist() if str(c).strip()])

# ---------- Exclusions ----------
with st.sidebar:
    # Sticky logo
    st.markdown('<div class="rcg-sidebar-logo">', unsafe_allow_html=True)
    try:
        st.image("logo.png", width=170)
    except Exception:
        pass
    st.markdown(f"<div style='text-align:center; font-weight:900; margin-top:6px;'>{APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Quick universal search
    st.markdown("### QUICK SEARCH")
    search_items = []
    for p in all_products:
        search_items.append(f"Product: {p}")
    for e in all_employees:
        search_items.append(f"Employee: {e}")
    for b in all_branches:
        search_items.append(f"Branch: {b}")
    for c in all_customers:
        search_items.append(f"Customer: {c}")
    search_pick = st.selectbox("Search", options=[""] + search_items, index=0, help="Type to search and press Enter.", key="universal_search")
    if search_pick:
        kind, val = search_pick.split(": ", 1)
        open_details(kind, val)

    st.markdown("---")
    remember_filters = st.toggle("Remember my filters", value=bool(SETTINGS.get("remember_filters", False)), key="remember_filters_toggle")
    SETTINGS["remember_filters"] = remember_filters
    save_settings(SETTINGS)

    if st.button("Reset ALL filters", key="reset_all_filters_btn"):
        st.session_state["__filters__"] = {
            "date_range": (dmin, dmax) if (dmin and dmax) else None,
            "branches": [],
            "employees": [],
            "channels": [],
            "depts": [],
            "customers": [],
            "products": [],
            "invoice": "",
            "returns_mode": "Include returns",
        }
        st.session_state["page"] = "Overview"
        st.rerun()

    st.markdown("---")
    st.markdown("---")
    st.header("INTERNAL CUSTOMERS")
    exclude_enabled = st.toggle("Exclude selected customers from stats", value=True)
    selected_to_exclude = st.multiselect(
        "Excluded customers",
        options=all_customers,
        default=[c for c in SETTINGS.get("excluded_customers", []) if c in all_customers],
        help="Exclusions are saved. Raw data is never changed.",
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("SAVE"):
            SETTINGS["excluded_customers"] = selected_to_exclude
            save_settings(SETTINGS)
            st.rerun()
    with b2:
        if st.button("CLEAR"):
            SETTINGS["excluded_customers"] = []
            save_settings(SETTINGS)
            st.rerun()

excluded = set(SETTINGS.get("excluded_customers", []))
df_stats = df.copy()
if exclude_enabled and excluded:
    df_stats = df_stats[~df_stats["Customer Name"].isin(excluded)]

# ---------- Filters ----------
sale_dates = df_stats["Sale Date"].dropna()
dmin = sale_dates.min().date() if not sale_dates.empty else None
dmax = sale_dates.max().date() if not sale_dates.empty else None

if "__filters__" not in st.session_state:
    base_filters = {
        "date_range": (dmin, dmax) if (dmin and dmax) else None,
        "branches": [],
        "employees": [],
        "channels": [],
        "depts": [],
        "customers": [],
        "products": [],
        "invoice": "",
        "returns_mode": "Include returns",
    }
    if SETTINGS.get("remember_filters") and isinstance(SETTINGS.get("last_filters"), dict):
        lf = SETTINGS.get("last_filters", {})
        for k in base_filters.keys():
            if k in lf and lf[k] is not None:
                base_filters[k] = lf[k]
    st.session_state["__filters__"] = base_filters

with st.sidebar:

    st.markdown("---")
    st.header("FILTERS")
    if dmin and dmax:
        dr = st.date_input("Sale date range", value=st.session_state["__filters__"]["date_range"], min_value=dmin, max_value=dmax)
        st.session_state["__filters__"]["date_range"] = dr
    else:
        dr = None
        st.warning("No valid Sale Date found.")

    st.session_state["__filters__"]["branches"] = st.multiselect("Branch", all_branches, default=st.session_state["__filters__"]["branches"])
    st.session_state["__filters__"]["employees"] = st.multiselect("Employee", all_employees, default=st.session_state["__filters__"]["employees"])
    st.session_state["__filters__"]["channels"] = st.multiselect("Channel", all_channels, default=st.session_state["__filters__"]["channels"])
    st.session_state["__filters__"]["depts"] = st.multiselect("Department", all_depts, default=st.session_state["__filters__"]["depts"])
    st.session_state["__filters__"]["customers"] = st.multiselect("Customer", all_customers, default=st.session_state["__filters__"]["customers"])
    st.session_state["__filters__"]["products"] = st.multiselect("Product", all_products, default=st.session_state["__filters__"]["products"])
    st.session_state["__filters__"]["invoice"] = st.text_input("Invoice contains", value=st.session_state["__filters__"]["invoice"])
    st.session_state["__filters__"]["returns_mode"] = st.radio(
        "Returns", ["Include returns","Exclude returns","Returns only"],
        index=["Include returns","Exclude returns","Returns only"].index(st.session_state["__filters__"]["returns_mode"])
    )


# Persist filters if enabled
if SETTINGS.get("remember_filters"):
    SETTINGS["last_filters"] = st.session_state["__filters__"]
    save_settings(SETTINGS)
def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    f = st.session_state["__filters__"]
    if f["date_range"] and len(f["date_range"]) == 2 and out["Sale Date"].notna().any():
        start, end = f["date_range"]
        out = out[out["Sale Date"].between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    if f["branches"]:
        out = out[out["Sale site"].isin(f["branches"])]
    if f["employees"]:
        out = out[out["Employee"].isin(f["employees"])]
    if f["channels"]:
        out = out[out["Channel"].isin(f["channels"])]
    if f["depts"]:
        out = out[out["Product Department"].isin(f["depts"])]
    if f["customers"]:
        out = out[out["Customer Name"].isin(f["customers"])]
    if f["products"]:
        out = out[out["Product Name"].isin(f["products"])]
    if f["invoice"].strip():
        s = f["invoice"].strip().lower()
        out = out[out["InvoiceReference"].astype(str).str.lower().str.contains(s, na=False)]
    if f["returns_mode"] == "Exclude returns":
        out = out[~out["Is Return"]]
    elif f["returns_mode"] == "Returns only":
        out = out[out["Is Return"]]
    return out

df_f = apply_filters(df_stats)
rank_col = "Total_Inc_VAT" if revenue_basis == "Total (incl VAT)" else "Total_Ex_VAT"

# ---------- KPIs ----------
inv = invoice_summary(df_f)
avg_units = inv["Net_Units"].mean() if len(inv) else np.nan

kpis = {
    "Total sales (inc VAT)": fmt_gbp(df_f["Total"].sum()),
    "Total sales (ex VAT)": fmt_gbp(df_f["Total Without Vat"].sum()),
    "Total profit": (fmt_gbp(df_f["Line Profit"].sum()) if not ignore_profit_everywhere else "—"),
    "VAT": fmt_gbp(df_f["VAT"].sum()),
    "Discounts": fmt_gbp(df_f["Discount"].sum()),
    "Restocking fees": fmt_gbp(df_f["Restocking Fee"].sum()),
    "Net units (sold - returned)": fmt_int(df_f["Quantity"].sum()),
    "Sold units": fmt_int(df_f.loc[df_f["Quantity"] > 0, "Quantity"].sum()),
    "Returned units": fmt_int(df_f.loc[df_f["Quantity"] < 0, "Quantity"].abs().sum()),
    "Sales lines": f"{len(df_f):,}",
    "Return lines": f"{int(df_f['Is Return'].sum()):,}",
    "Avg units per invoice": (f"{avg_units:,.1f}" if pd.notna(avg_units) else "—"),
}

r1 = st.columns(5)
for i, (k, v) in enumerate(list(kpis.items())[:5]):
    r1[i].metric(k, v)
r2 = st.columns(5)
for i, (k, v) in enumerate(list(kpis.items())[5:10]):
    r2[i].metric(k, v)

st.markdown("---")


# ---------- Navigation ----------
PAGES = ["Overview","Branches","Employees","Products","Customers","Returns","Invoices","Reports","Details","Raw data","Export"]
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"
page = st.radio("Navigate", options=PAGES, index=PAGES.index(st.session_state["page"]), horizontal=True, label_visibility="collapsed")
st.session_state["page"] = page


# Active filters bar
f = st.session_state["__filters__"]
chips = []
if f.get("date_range") and len(f["date_range"])==2:
    chips.append(f"Dates: {f['date_range'][0]} → {f['date_range'][1]}")
if f.get("branches"):
    chips.append("Branches: " + ", ".join(map(str,f["branches"][:3])) + ("…" if len(f["branches"])>3 else ""))
if f.get("employees"):
    chips.append("Employees: " + ", ".join(map(str,f["employees"][:3])) + ("…" if len(f["employees"])>3 else ""))
if f.get("products"):
    chips.append("Products: " + ", ".join(map(str,f["products"][:2])) + ("…" if len(f["products"])>2 else ""))
if exclude_enabled and excluded:
    chips.append(f"Internal customers excluded: {len(excluded)}")
chips.append("Revenue: inc VAT" if revenue_basis=="Total (incl VAT)" else "Revenue: ex VAT")
chips.append("Profit: ignored" if ignore_profit_everywhere else "Profit: included")
st.markdown("**Active filters:** " + " | ".join(chips))

# ---------- Aggregations ----------
by_branch = summarize_group(df_f, "Sale site").sort_values(rank_col, ascending=False)
by_employee = summarize_group(df_f, "Employee").sort_values(rank_col, ascending=False)
by_product = product_rollup(df_f).sort_values(rank_col, ascending=False)
by_customer = summarize_group(df_f, "Customer Name").sort_values(rank_col, ascending=False)
by_returns = product_rollup(df_f[df_f["Is Return"]].copy()).sort_values(rank_col, ascending=False)

# Trend
if df_f["Sale Date"].notna().any():
    tr = df_f.copy()
    tr["Sale Day"] = df_f["Sale Date"].dt.date
    by_day = tr.groupby("Sale Day", observed=True).agg(Total_Inc_VAT=("Total","sum"), Total_Ex_VAT=("Total Without Vat","sum")).reset_index().sort_values("Sale Day")
else:
    by_day = pd.DataFrame(columns=["Sale Day","Total_Inc_VAT","Total_Ex_VAT"])

# ---------- Details selection state ----------
if "details" not in st.session_state:
    st.session_state["details"] = {"kind": None, "value": None, "apply_filters": True}

def open_details(kind: str, value: str):
    apply_choice = details_context_prompt()
    st.session_state["details"] = {"kind": kind, "value": value, "apply_filters": apply_choice}
    st.session_state["page"] = "Details"
    remember_recent(kind, value)
    st.rerun()

# ---------- Tabs ----------

money_cols_main = ["Total_Inc_VAT","Total_Ex_VAT","VAT","Discount"]
int_cols_main = ["Sales_Lines","Net_Units","Return_Lines"]

def detail_df(kind: str, value: str, apply_current: bool) -> pd.DataFrame:
    base = df_f if apply_current else df_stats
    if kind == "Product":
        return base[base["Product Name"].astype(str) == value]
    if kind == "Employee":
        return base[base["Employee"].astype(str) == value]
    if kind == "Branch":
        return base[base["Sale site"].astype(str) == value]
    if kind == "Customer":
        return base[base["Customer Name"].astype(str) == value]
    return base.iloc[0:0]

def detail_kpis(df_in: pd.DataFrame) -> Dict[str, str]:
    inv2 = invoice_summary(df_in)
    return {
        "Revenue (inc VAT)": fmt_gbp(df_in["Total"].sum()),
        "Revenue (ex VAT)": fmt_gbp(df_in["Total Without Vat"].sum()),
        "Net units": fmt_int(df_in["Quantity"].sum()),
        "Sold units": fmt_int(df_in.loc[df_in["Quantity"] > 0, "Quantity"].sum()),
        "Returned units": fmt_int(df_in.loc[df_in["Quantity"] < 0, "Quantity"].abs().sum()),
        "Invoices": f"{len(inv2):,}",
        "Sales lines": f"{len(df_in):,}",
        "Return lines": f"{int(df_in['Is Return'].sum()):,}",
    }

def show_transactions(df_in: pd.DataFrame, label: str):
    st.subheader("Transactions")
    cols = ["Sale Date","InvoiceReference","Sale site","Employee","Customer Name","Product Name","Quantity","Total","Total Without Vat","Discount","Is Return"]
    view = df_in.copy()
    for c in cols:
        if c not in view.columns:
            view[c] = np.nan
    view = view[cols]

    show_all = st.toggle("Show all rows (slower)", value=False, key=f"showall_{label}")
    view2 = view if show_all else view.head(500)
    st.dataframe(view2, use_container_width=True, hide_index=True)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export transactions CSV", csv, f"{label}_transactions.csv", "text/csv")


# ---------- Reports ----------
def report_sales_on_misc(df_stats: pd.DataFrame, revenue_col: str, date_range=None):
    """
    Sales on MISC report:
    Includes customer names that are:
      - blank / missing
      - MISC2
      - Mr MISC2
      - Mr MISC2 MISC2
      - Mr Misc Misc2
    Case-insensitive, spacing-insensitive.
    Removes rows where total value == 0 (both view + export).
    Count is unique Sale Reference.
    """
    import pandas as pd

    if date_range is None:
        date_range = _last_full_month()

    d0, d1 = date_range
    dff = df_stats.copy()

    # Date filter uses Sale Date when present
    if "Sale Date" in dff.columns:
        dff["Sale Date"] = pd.to_datetime(dff["Sale Date"], errors="coerce")
        dff = dff[dff["Sale Date"].notna()].copy()
        dff = dff[(dff["Sale Date"].dt.date >= d0) & (dff["Sale Date"].dt.date <= d1)].copy()

    # Customer normalization (safe for Categoricals)
    cust = dff.get("Customer Name", pd.Series([""] * len(dff)))
    cust = cust.astype("string").fillna("")
    cust_norm = cust.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

    # Blank customers
    is_blank = cust_norm.eq("") | cust_norm.isin(["nan", "none", "null"])

    # MISC2 variants (regex after normalization)
    misc_re = r"^(mr\s+)?(misc\s+)?misc2(\s+misc2)?$"
    is_misc = cust_norm.str.match(misc_re, na=False)

    dff = dff[is_blank | is_misc].copy()

    if dff.empty:
        empty_emp = pd.DataFrame(columns=["Employee", "Total_Value", "Sales_Count"])
        empty_br = pd.DataFrame(columns=["Sale site", "Total_Value", "Sales_Count"])
        return dff, empty_emp, empty_br, {"grand_total": 0.0, "grand_count": 0, "d0": d0, "d1": d1}

    # Ensure keys exist
    if "Employee" not in dff.columns:
        dff["Employee"] = "Unknown"
    if "Sale site" not in dff.columns and "Branch" in dff.columns:
        dff["Sale site"] = dff["Branch"]
    if "Sale site" not in dff.columns:
        dff["Sale site"] = "Unknown"
    if "Sale Reference" not in dff.columns:
        dff["Sale Reference"] = ""

    dff["Employee"] = dff["Employee"].astype("string").fillna("Unknown")
    dff["Sale site"] = dff["Sale site"].astype("string").fillna("Unknown")
    dff["Sale Reference"] = dff["Sale Reference"].astype("string").fillna("")

    # Revenue numeric
    if revenue_col not in dff.columns:
        dff[revenue_col] = 0.0
    dff[revenue_col] = pd.to_numeric(dff[revenue_col], errors="coerce").fillna(0.0)

    emp = (
        dff.groupby("Employee", dropna=False)
        .agg(Total_Value=(revenue_col, "sum"), Sales_Count=("Sale Reference", pd.Series.nunique))
        .reset_index()
    )
    br = (
        dff.groupby("Sale site", dropna=False)
        .agg(Total_Value=(revenue_col, "sum"), Sales_Count=("Sale Reference", pd.Series.nunique))
        .reset_index()
    )

    emp["Total_Value"] = pd.to_numeric(emp["Total_Value"], errors="coerce").fillna(0.0)
    br["Total_Value"] = pd.to_numeric(br["Total_Value"], errors="coerce").fillna(0.0)

    # Drop £0 rows
    emp = emp.loc[emp["Total_Value"] != 0].copy().sort_values("Total_Value", ascending=False)
    br = br.loc[br["Total_Value"] != 0].copy().sort_values("Total_Value", ascending=False)

    grand_total = float(br["Total_Value"].sum()) if not br.empty else 0.0
    grand_count = int(emp["Sales_Count"].sum()) if not emp.empty else 0

    return dff, emp, br, {"grand_total": grand_total, "grand_count": grand_count, "d0": d0, "d1": d1}


def drill_tab(kind: str, df_rank: pd.DataFrame, key_col: str, options: List[str], select_key: str):
    st.subheader(f"{kind}s")
    sel = selection_block(kind, options, select_key)
    if st.button(f"View {kind} details", disabled=not bool(sel), key=f"btn_{select_key}"):
        open_details(kind, sel)

    c1, c2 = st.columns([1.25, 1])
    with c1:
        if not df_rank.empty:
            choose_chart_money(df_rank, top_chart, key_col,
                               "Total_Inc_VAT" if revenue_basis == "Total (incl VAT)" else "Total_Ex_VAT",
                               f"Top {kind}s", y_is_money=True)
        else:
            st.info("No rows after filters.")
    with c2:
        st.dataframe(style_table(df_rank.head(200), money_cols_main, int_cols_main), use_container_width=True, hide_index=True)

if page == "Branches":
    drill_tab("Branch", by_branch, "Sale site", all_branches, "br")

if page == "Employees":
    drill_tab("Employee", by_employee, "Employee", all_employees, "emp")

if page == "Products":
    st.subheader("Products")
    sel = selection_block("Product", all_products, "prod")
    if st.button("View product details", disabled=not bool(sel), key="btn_view_product_details_products"):
        open_details("Product", sel)
    st.dataframe(style_table(by_product.head(200)[["Product","Department","Sales_Lines","Net_Units","Total_Inc_VAT","Total_Ex_VAT","Return_Lines"]],
                             ["Total_Inc_VAT","Total_Ex_VAT"], ["Sales_Lines","Net_Units","Return_Lines"]),
                 use_container_width=True, hide_index=True)

if page == "Customers":
    drill_tab("Customer", by_customer, "Customer Name", all_customers, "cust")

if page == "Returns":
    st.subheader("Returns")
    if by_returns.empty:
        st.info("No returns in the filtered dataset.")
    else:
        choose_chart_money(by_returns.head(20), top_chart, "Product",
                           "Total_Inc_VAT" if revenue_basis == "Total (incl VAT)" else "Total_Ex_VAT",
                           "Top Returned Products", y_is_money=True)
        st.dataframe(style_table(by_returns.head(200)[["Product","Sales_Lines","Net_Units","Total_Inc_VAT","Total_Ex_VAT","Return_Lines"]],
                                 ["Total_Inc_VAT","Total_Ex_VAT"], ["Sales_Lines","Net_Units","Return_Lines"]),
                     use_container_width=True, hide_index=True)

if page == "Invoices":
    st.subheader("Invoices")
    inv2 = inv.sort_values("Total_Inc_VAT" if revenue_basis == "Total (incl VAT)" else "Total_Ex_VAT", ascending=False)
    st.dataframe(style_table(inv2.head(500), ["Total_Inc_VAT","Total_Ex_VAT"], ["Sales_Lines","Net_Units"]), use_container_width=True, hide_index=True)


if page == "Reports":
    st.subheader("Reports")
    st.caption("Generate reports. Defaults to last full month; adjust as needed.")

    if "report_selected" not in st.session_state:
        st.session_state["report_selected"] = "misc"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Sales on MISC")
        st.caption("MISC2 variants + blank customer names. Count = unique Sale Reference.")
        if st.button("View report", key="rep_misc_view"):
            st.session_state["report_selected"] = "misc"
            st.rerun()
    with c2:
        st.markdown("#### Coming soon")
        st.caption("Add more retail reports here.")
    with c3:
        st.markdown("#### Coming soon")
        st.caption("Add more retail reports here.")

    st.markdown("---")

    if st.session_state["report_selected"] == "misc":
        st.markdown("## Sales on MISC")
        d0_def, d1_def = _last_full_month_range()
        r1, r2, r3 = st.columns([1,1,1.2])
        with r1:
            d0 = st.date_input("From", value=d0_def, key="misc_d0")
        with r2:
            d1 = st.date_input("To", value=d1_def, key="misc_d1")
        with r3:
            st.write("")
            st.caption(f"Revenue basis: **{revenue_basis}**")

        revenue_col = "Total" if revenue_basis == "Total (incl VAT)" else "Total Without Vat"
        misc_df, emp, br, meta = report_sales_on_misc(df_f, revenue_col=revenue_col, date_range=(d0, d1))

        m1, m2 = st.columns(2)
        m1.metric("Total MISC revenue", fmt_gbp(meta["grand_total"]))
        m2.metric("Sales count (unique Sale Reference)", f"{meta['grand_count']:,}")

        st.markdown("### By employee")
        if emp.empty:
            st.info("No MISC sales for the selected period.")
        else:
            choose_chart_money(emp, top_chart, "Employee", "Total_Value", "All employees", y_is_money=True)
            st.dataframe(style_table(emp, ["Total_Value"], ["Sales_Count"]), use_container_width=True, hide_index=True)

        st.markdown("### By branch")
        if br.empty:
            st.info("No branches with non-zero MISC sales for the selected period.")
        else:
            choose_chart_money(br, top_chart, "Sale site", "Total_Value", "All branches", y_is_money=True)
            st.dataframe(style_table(br, ["Total_Value"], ["Sales_Count"]), use_container_width=True, hide_index=True)

        st.markdown("### Download")
        tables = {
            "MISC By Employee": emp.rename(columns={"Total_Value":"Total"}),
            "MISC By Branch": br.rename(columns={"Total_Value":"Total"}),
        }
        summary_kpis = {
            "From": str(d0),
            "To": str(d1),
            "Revenue basis": revenue_basis,
            "Total MISC revenue": fmt_gbp(meta["grand_total"]),
            "Sales count": f"{meta['grand_count']:,}",
        }
        excel_bytes = make_excel_report(misc_df, summary_kpis, tables)
        st.download_button(
            "⬇️ Download Sales on MISC (Excel)",
            excel_bytes,
            file_name=f"Sales_on_MISC_{d0}_{d1}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_misc_excel",
        )

if page == "Details":
    st.subheader("Details")
    det = st.session_state.get("details", {})
    kind = det.get("kind")
    value = det.get("value")
    apply_current = bool(det.get("apply_filters", True))

    rec = SETTINGS.get("recent", {})
    if rec:
        with st.expander("Recently viewed", expanded=False):
            for k in ["Product","Employee","Branch","Customer"]:
                items = rec.get(k, [])
                if items:
                    st.caption(k)
                    for it in items:
                        if st.button(f"Open {k}: {it}", key=f"rec_{k}_{it}"):
                            st.session_state["details"] = {"kind": k, "value": it, "apply_filters": apply_current}
                            st.session_state["page"] = "Details"
                            st.rerun()

    if not (kind and value):
        st.info("Use 'View details' from any tab to open an item here.")
    else:
        st.markdown(f"### {kind}: **{value}**")
        st.caption("Net units = sold − returns. Revenue defaults to £ incl VAT (toggle in sidebar).")
        ddf = detail_df(kind, value, apply_current)
        if ddf.empty:
            st.warning("No rows for this item under the chosen scope/filters.")
        else:
            dk = detail_kpis(ddf)
            cols = st.columns(4)
            for i, (k, v) in enumerate(dk.items()):
                cols[i % 4].metric(k, v)

            st.markdown("---")
            st.subheader("Breakdowns")
            b1, b2 = st.columns(2)
            with b1:
                st.caption("By branch")
                tb = summarize_group(ddf, "Sale site").sort_values(rank_col, ascending=False).head(20)
                st.dataframe(style_table(tb, money_cols_main, int_cols_main), use_container_width=True, hide_index=True)
            with b2:
                st.caption("By employee")
                te = summarize_group(ddf, "Employee").sort_values(rank_col, ascending=False).head(20)
                st.dataframe(style_table(te, money_cols_main, int_cols_main), use_container_width=True, hide_index=True)

            st.caption("By date")
            if ddf["Sale Date"].notna().any():
                t = ddf.copy()
                t["Sale Day"] = ddf["Sale Date"].dt.date
                td = t.groupby("Sale Day", observed=True).agg(Total_Inc_VAT=("Total","sum"), Total_Ex_VAT=("Total Without Vat","sum")).reset_index().sort_values("Sale Day")
                choose_chart_money(td, trend_chart, "Sale Day",
                                   "Total_Inc_VAT" if revenue_basis == "Total (incl VAT)" else "Total_Ex_VAT",
                                   "Revenue over time (detail)", y_is_money=True)
            else:
                st.info("No valid dates for trend.")

            st.markdown("---")
            safe = re.sub(r'[^A-Za-z0-9]+','_', value)[:40]
            show_transactions(ddf.sort_values("Sale Date", ascending=False), f"{kind}_{safe}")

            st.subheader("Detail export")
            tables = {
                "By Branch": summarize_group(ddf, "Sale site").sort_values(rank_col, ascending=False),
                "By Employee": summarize_group(ddf, "Employee").sort_values(rank_col, ascending=False),
                "By Customer": summarize_group(ddf, "Customer Name").sort_values(rank_col, ascending=False).head(200),
            }
            excel_bytes = make_excel_report(ddf, dk, tables)
            st.download_button("⬇️ Export detail report (Excel)", excel_bytes, f"{kind}_{safe}_Report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if page == "Raw data":
    st.subheader("Raw data (not affected by exclusions)")
    st.caption("Showing first 500 rows for speed.")
    st.dataframe(df.head(500), use_container_width=True, hide_index=True)

if page == "Export":
    st.subheader("Current filter pack export")
    inv2 = inv.sort_values("Total_Inc_VAT" if revenue_basis == "Total (incl VAT)" else "Total_Ex_VAT", ascending=False)
    tables = {"Branches": by_branch, "Employees": by_employee, "Products": by_product, "Customers": by_customer, "Returns": by_returns, "Invoices": inv2}
    excel_bytes = make_excel_report(df_f, kpis, tables)
    st.download_button("⬇️ Export Excel pack", excel_bytes, "YJ-RCG_Current_Filter_Pack.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
