import streamlit as st
import pandas as pd
import numpy as np
import time
import eda_utils
import search_service
from datetime import datetime
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

# --- MONKEYPATCH: Fix for 'asarray() got an unexpected keyword argument copy' ---
# ... (keeping it for safety)
_original_asarray = np.asarray
def _patched_asarray(a, dtype=None, order=None, *, copy=None, **kwargs):
    if copy is True:
        return np.array(a, dtype=dtype, order=order, copy=True)
    return _original_asarray(a, dtype=dtype, order=order)
np.asarray = _patched_asarray
# --------------------------------------------------------------------------------

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Kaggle Scraper & Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching the Profiling Report ---
@st.cache_resource
def generate_profile_html(df):
    """Generates the YData Profiling report as HTML."""
    profile = ProfileReport(df, explorative=True, title="Data Profile Report")
    return profile.to_html()

# --- Custom Styling ---
def apply_custom_styling():
    # ... (same as before)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .stApp { font-family: 'Inter', sans-serif; }
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            padding: 2rem; border-radius: 12px; text-align: center; color: white; margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; color: white !important; }
        .main-header p { font-size: 1rem; opacity: 0.9; color: #e2e8f0 !important; }
        .custom-card {
            background-color: var(--secondary-background-color);
            padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.1); margin-bottom: 1.5rem;
        }
        .section-header {
            font-size: 1.2rem; font-weight: 600; color: var(--text-color); margin-bottom: 1rem;
            padding-bottom: 0.5rem; border-bottom: 2px solid #3b82f6; display: flex; align-items: center; gap: 0.5rem;
        }
        .metric-container {
            background-color: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; text-align: center;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        .metric-icon { font-size: 1.5rem; margin-bottom: 0.25rem; }
        .metric-value { font-size: 1.5rem; font-weight: 700; color: #3b82f6; }
        .metric-label { font-size: 0.85rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-color); }
        .stButton > button { background-color: #2563eb; color: white; border: none; border-radius: 6px; font-weight: 500; transition: background-color 0.2s; }
        .stButton > button:hover { background-color: #1d4ed8; color: white; }
        div.block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(label, value, icon="📊"):
    return f"""
    <div class="metric-container">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

# --- Callback ---
def handle_analyze_callback(df, title):
    """Callback to switch to Analyze tab and load data."""
    st.session_state['current_data'] = df
    st.session_state['dataset_title'] = title
    st.session_state['navigation'] = "Upload & Analyze"

# --- Main App Logic ---
def main():
    apply_custom_styling()

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Kaggle Scraper & Analyzer</h1>
        <p>Enterprise Data Discovery & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Session State ---
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'dataset_title' not in st.session_state:
        st.session_state['dataset_title'] = "Uploaded Data"
    if 'navigation' not in st.session_state:
        st.session_state['navigation'] = "Search & Scrape"

    # --- Sidebar ---
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Navigation</div>', unsafe_allow_html=True)
        mode = st.radio("Choose Mode:", ["Search & Scrape", "Upload & Analyze"], key="navigation", label_visibility="collapsed")
        st.markdown("---")
        st.info("💡 **Tip:** Start by searching for a topic, then click 'Analyze' to visualize the results.")
        
        # Credentials check
        if not search_service.get_kaggle_api():
            st.error("⚠️ Kaggle API not configured. Please ensure `kaggle.json` is in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.")

    # --- Mode A: Search & Scrape ---
    if mode == "Search & Scrape":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 Intelligent Search</div>', unsafe_allow_html=True)
        
        query = st.text_input("Describe your dataset:", placeholder="e.g. 'Student exam scores', 'Population of China'...")
        
        if st.button("🚀 Search Kaggle", type="primary"):
            if query:
                with st.status("🔍 AI is processing your request...") as status:
                    st.write("🤖 Extracting requirements with LLM...")
                    query_info = search_service.extract_dataset_request(query)
                    if query_info:
                        st.session_state['query_info'] = query_info
                        topic = query_info.get('topic', query)
                        st.write(f"🔎 Searching Kaggle for: `{topic}`...")
                        results = search_service.kaggle_search(topic)
                        st.session_state['search_results'] = results
                        status.update(label="✅ Search complete!", state="complete")
                    else:
                        st.error("Failed to extract info from query.")
                        status.update(label="❌ Failed", state="error")
            else:
                st.warning("Please enter a query.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display Results
        if 'search_results' in st.session_state and st.session_state['search_results']:
            st.markdown("### 📚 Found Datasets")
            for idx, res in enumerate(st.session_state['search_results']):
                with st.expander(f"📄 {res['title']} ({res['ref']})"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Dataset Reference: `{res['ref']}`")
                    with col2:
                        if st.button("📥 Load & Evaluate", key=f"load_{idx}"):
                            with st.spinner("Downloading and validating..."):
                                df, error = search_service.download_and_load_dataset(res['ref'])
                                if df is not None:
                                    # Validate with LLM
                                    validation = search_service.validate_dataset_with_llm(
                                        df, st.session_state['query_info'], query, res['title']
                                    )
                                    
                                    st.success("Dataset loaded!")
                                    st.markdown(f"**Relevance Score:** {validation['relevance_score']*100:.0f}%")
                                    st.info(f"**Reasoning:** {validation['reasoning']}")
                                    st.write(f"**Use Cases:** {validation['use_cases']}")
                                    
                                    st.dataframe(df.head(), use_container_width=True)
                                    
                                    st.button("📊 Go to Analysis Dashboard", 
                                              on_click=handle_analyze_callback, 
                                              args=(df, res['title']),
                                              key=f"analyze_{idx}")
                                else:
                                    st.error(f"Error: {error}")
        elif 'search_results' in st.session_state:
            st.info("No datasets found matching that topic.")

    # --- Mode B: Upload & Analyze ---
    elif mode == "Upload & Analyze":
        # ... (rest of the code remains similar)
        
        # Data Loading Section
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📁 Data Source</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
        
        df = None
        if uploaded_file is not None:
            df = eda_utils.load_data(uploaded_file)
            st.session_state['dataset_title'] = uploaded_file.name
        elif st.session_state['current_data'] is not None:
            df = st.session_state['current_data']
            st.markdown(f"**Using Active Dataset:** `{st.session_state['dataset_title']}`")
        else:
            st.info("👆 Please upload a file or search for a dataset first.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dashboard Section
        if df is not None:
            stats = eda_utils.get_quick_stats(df)
            
            # Quick Stats Row with Custom Cards
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚡ Quick Statistics</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(create_metric_card("Rows", stats["rows"], "🔢"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_metric_card("Columns", stats["cols"], "📊"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_metric_card("Missing", f"{stats['missing_pct']}%", "❓"), unsafe_allow_html=True)
            with col4:
                st.markdown(create_metric_card("Duplicates", stats["duplicates"], "👯"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "🔥 Correlations", "📄 Raw Data", "📑 Auto-Insight Report"])
            
            num_cols, cat_cols = eda_utils.detect_column_types(df)
            
            with tab1:
                st.markdown("### Variable Distributions")
                all_cols = num_cols + cat_cols
                if all_cols:
                    selected_col = st.selectbox("Select Column:", all_cols)
                    chart = eda_utils.plot_distributions(df, selected_col)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("No suitable columns found.")
            
            with tab2:
                st.markdown("### Correlation Matrix")
                if len(num_cols) > 1:
                    corr_chart = eda_utils.plot_correlation(df)
                    if corr_chart:
                        st.plotly_chart(corr_chart, use_container_width=True)
                else:
                    st.info("Not enough numerical columns for correlation.")
            
            with tab3:
                st.markdown("### Data Browser")
                st.dataframe(df, use_container_width=True)

            with tab4:
                st.markdown("### 🔍 Deep Dive Automated Analysis")
                st.info("Generating comprehensive data profile... This may take a moment.")
                try:
                    # Generate the HTML report
                    profile_html = generate_profile_html(df)
                    # Display using Streamlit components
                    components.html(profile_html, height=1000, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to generate profiling report: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
