import streamlit as st

PAGE_CONFIG = {
    "page_title": "智水先知 V16.0 | 企业版",
    "page_icon": "💧",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# 企业级浅色工业风 CSS — 清晰明亮，高对比度
CUSTOM_CSS = """
<style>
    /* ===== 全局 ===== */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

    html, body {
        color-scheme: light;
    }

    .stApp {
        background-color: #eef1f6 !important;
        color: #1a1a1a !important;
        font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
    }

    /* ===== 侧边栏 ===== */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dde2ec !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1677ff !important;
    }

    /* ===== 顶部栏 ===== */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ===== 标题文字 ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-family: 'Noto Sans SC', sans-serif !important;
    }

    /* ===== 横幅标题（app.py 里的 header）====== */
    .main-title-banner {
        background: linear-gradient(135deg, #1677ff 0%, #4096ff 100%);
        color: #ffffff !important;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 12px rgba(22,119,255,0.3);
    }
    .main-title-banner h1 {
        color: #ffffff !important;
        font-size: 22px;
        margin: 0;
    }
    .main-title-banner p {
        color: rgba(255,255,255,0.9) !important;
        margin: 4px 0 0;
        font-size: 13px;
    }

    /* ===== 卡片 ===== */
    .hud-card {
        background: #ffffff !important;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid #dde2ec !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        color: #1a1a1a !important;
    }
    .hud-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    /* ===== 图表专用卡片（白底+清晰边框）====== */
    .hud-card-chart {
        background: #f8fafd !important;
        border-radius: 10px;
        padding: 20px 20px 12px;
        margin-bottom: 16px;
        border: 1px solid #dde2ec !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        color: #1a1a1a !important;
    }

    /* ===== KPI 指标卡片 (st.metric) ===== */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border-radius: 8px !important;
        padding: 16px 12px !important;
        border: 1px solid #dde2ec !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
        text-align: left !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 30px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #888888 !important;
        font-size: 12px !important;
    }
    /* st.caption under metric */
    [data-testid="stMetric"] + [data-testid="stCaptionContainer"] {
        color: #555555 !important;
        font-size: 12px !important;
    }

    /* ===== 按钮 ===== */
    .stButton > button {
        width: 100%;
        background-color: #1677ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        height: 42px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #4096ff !important;
        box-shadow: 0 4px 12px rgba(22,119,255,0.35) !important;
        color: #ffffff !important;
    }
    .stButton > button:active {
        background-color: #0958d9 !important;
        color: #ffffff !important;
    }

    /* ===== 下拉框（Selectbox — 侧边栏内）====== */
    /* 侧边栏内的 selectbox */
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stSelectbox"] > div {
        background-color: #f5f7fa !important;
        border: 1px solid #c8d0db !important;
        border-radius: 6px !important;
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #555555 !important;
    }
    /* 下拉选项菜单 */
    ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #c8d0db !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
        border-radius: 6px !important;
        padding: 4px !important;
    }
    li[role="option"] {
        color: #1a1a1a !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }
    li[role="option"]:hover {
        background-color: #e6f0ff !important;
        color: #1677ff !important;
    }
    li[aria-selected="true"] {
        background-color: #ddeaff !important;
        color: #1677ff !important;
        font-weight: 600 !important;
    }

    /* ===== Slider ===== */
    [data-testid="stSlider"] label {
        color: #1a1a1a !important;
    }
    [data-testid="stSlider"] span {
        color: #555555 !important;
    }

    /* ===== Checkbox ===== */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] span,
    [data-testid="stCheckbox"] p {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }

    /* ===== 输入框 ===== */
    [data-testid="stTextInput"] label {
        color: #1a1a1a !important;
    }
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #c8d0db !important;
        border-radius: 6px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1677ff !important;
        box-shadow: 0 0 0 2px rgba(22,119,255,0.15) !important;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-bottom: 2px solid #dde2ec !important;
        border-radius: 8px 8px 0 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #666666 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1677ff !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1677ff !important;
        font-weight: 700 !important;
        border-bottom-color: #1677ff !important;
    }

    /* ===== 下载按钮 ===== */
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #c8d0db !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #f0f5ff !important;
        border-color: #1677ff !important;
        color: #1677ff !important;
    }

    /* ===== Alert 信息提示 ===== */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 16px !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {
        color: #ffffff !important;
    }

    /* ===== Dataframe / Table ===== */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border-color: #dde2ec !important;
        font-size: 13px !important;
    }
    [data-testid="stDataFrame"] thead th {
        background-color: #f5f7fa !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        background-color: #f0f5ff !important;
    }

    /* ===== Plotly 图表 ===== */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 6px !important;
    }

    /* 强制 Plotly 图表 SVG 显示，防止被遮挡 */
    [data-testid="stPlotlyChart"] {
        display: block !important;
        width: 100% !important;
        min-height: 460px !important;
    }
    [data-testid="stPlotlyChart"] > div {
        width: 100% !important;
    }
    [data-testid="stPlotlyChart"] iframe {
        width: 100% !important;
        min-height: 460px !important;
    }
    /* 确保 element-container 不限制图表尺寸 */
    [data-testid="element-container"] {
        width: 100% !important;
        max-width: none !important;
    }

    /* ===== Divider ===== */
    hr {
        border-color: #dde2ec !important;
    }

    /* ===== Caption / Help text ===== */
    .stCaption, .stHelpText {
        color: #666666 !important;
        font-size: 12px !important;
    }

    /* ===== Markdown text ===== */
    .stMarkdown p,
    .stMarkdown div {
        color: #1a1a1a !important;
    }
    .stMarkdown strong {
        color: #1a1a1a !important;
    }

    /* ===== Progress bar ===== */
    .stProgress > div > div > div {
        background-color: #1677ff !important;
    }

    /* ===== block-container padding ===== */
    .block-container {
        padding-top: 1rem;
    }

    /* ===== RAG info box (知识问答回答) ===== */
    [data-testid="stAlert"][class*="info"] {
        background-color: rgba(22,119,255,0.12) !important;
        border-left: 4px solid #1677ff !important;
        color: #1a1a1a !important;
    }
    [data-testid="stAlert"][class*="info"] p,
    [data-testid="stAlert"][class*="info"] span {
        color: #1a1a1a !important;
    }

    /* ===== 数字孪生推演标题卡片 ===== */
    .twin-matrix-card {
        background: #ffffff !important;
        border-radius: 10px !important;
        padding: 16px !important;
        border: 1px solid #dde2ec !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    }
    .twin-matrix-card h3,
    .twin-matrix-card h4 {
        color: #1a1a1a !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
</style>
"""
