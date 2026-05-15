# app.py - Production Ready IDD Credit Scoring Engine
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import pickle

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="IDD Credit Scoring",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS for Professional Production Ready UI
# =============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card styling */
    .card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #1a3c5e 0%, #2c5a7a 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-value {
        font-size: 4rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
        line-height: 1;
    }
    .score-label {
        font-size: 0.875rem;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .risk-low {
        background-color: #d1fae5;
        color: #065f46;
    }
    .risk-medium {
        background-color: #fed7aa;
        color: #92400e;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Metric boxes */
    .metric-box {
        background-color: #f9fafb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .score-value {
            font-size: 2.5rem;
        }
        .metric-value {
            font-size: 1.25rem;
        }
        .card {
            padding: 1rem;
        }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f9fafb;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Info box */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FICOScorer Class Definition
# =============================================================================

class FICOScorer:
    VERSION = "4.0.0"
    
    CLUSTER_FEATURES = [
        "num_farms", "land_size_log", "avg_land_log", "crop_diversity",
        "multi_farm", "staple_crop_avg_freq", "is_diversified", "primary_crop_freq",
    ]
    
    SCORE_WEIGHTS = {
        "total_farm_scale": 0.30,
        "land_quality": 0.15,
        "geographic_spread": 0.10,
        "crop_diversity": 0.15,
        "staple_reliability": 0.10,
        "market_liquidity": 0.20,
    }
    
    def __init__(self, scaler, kmeans, cluster_to_tier, training_distributions, score_min, score_max):
        self.scaler = scaler
        self.kmeans = kmeans
        self.cluster_to_tier = cluster_to_tier
        self.training_distributions = training_distributions
        self.score_min = score_min
        self.score_max = score_max
    
    def _pct_rank(self, values, ref):
        if ref is None or len(ref) == 0:
            return np.full(len(values), 50.0)
        return np.searchsorted(ref, np.nan_to_num(values, nan=0.0), side="left") / len(ref) * 100
    
    def predict(self, df, include_contributions=False):
        df = df.copy()
        X_scaled = self.scaler.transform(df[self.CLUSTER_FEATURES])
        df["cluster"] = self.kmeans.predict(X_scaled)
        df["cluster_label"] = df["cluster"].map(self.cluster_to_tier)
        
        raw = np.zeros(len(df))
        for feat, weight_name in [
            ("land_size_log", "total_farm_scale"),
            ("avg_land_log", "land_quality"),
            ("num_farms", "geographic_spread"),
            ("crop_diversity", "crop_diversity"),
            ("staple_crop_avg_freq", "staple_reliability"),
            ("primary_crop_freq", "market_liquidity")
        ]:
            ref = self.training_distributions.get(feat, np.array([0, 100]))
            pct = self._pct_rank(df[feat].values, ref)
            raw += pct * self.SCORE_WEIGHTS[weight_name]
        
        span = (self.score_max - self.score_min) + 1e-9
        df["idd_score"] = (300 + (raw - self.score_min) / span * 550).round(0).astype(int).clip(300, 850)
        df["risk_tier"] = df["cluster_label"]
        
        return df[["cluster", "cluster_label", "idd_score", "risk_tier"]]

# =============================================================================
# Load Model
# =============================================================================
@st.cache_resource
def load_model():
    MODEL_PATH = Path("artifacts/fico_scorer.pkl")
    try:
        return joblib.load(MODEL_PATH)
    except:
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            st.warning("Demo mode active")
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            kmeans = KMeans(n_clusters=3, random_state=42)
            training_distributions = {
                "land_size_log": np.array([0, 1, 2, 3, 4, 5]),
                "avg_land_log": np.array([0, 1, 2, 3, 4, 5]),
                "num_farms": np.array([1, 2, 3, 4, 5]),
                "crop_diversity": np.array([1, 2, 3, 4, 5]),
                "staple_crop_avg_freq": np.array([0, 0.25, 0.5, 0.75, 1]),
                "primary_crop_freq": np.array([0, 0.25, 0.5, 0.75, 1]),
            }
            return FICOScorer(scaler, kmeans, {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"},
                            training_distributions, 0, 100)

scorer = load_model()

# =============================================================================
# Helper Functions
# =============================================================================
STAPLE_CROPS = {'Maize', 'Teff', 'Barley', 'Wheat'}

def calculate_features(total_land, num_farms, crops):
    avg_land = total_land / num_farms if num_farms > 0 else 0
    crop_diversity = len(set(crops))
    staple_count = sum(1 for c in crops if c in STAPLE_CROPS)
    staple_ratio = staple_count / len(STAPLE_CROPS) if len(STAPLE_CROPS) > 0 else 0
    
    return {
        "num_farms": float(num_farms),
        "land_size_log": np.log1p(total_land),
        "avg_land_log": np.log1p(avg_land),
        "crop_diversity": float(crop_diversity),
        "multi_farm": 1.0 if num_farms > 1 else 0.0,
        "staple_crop_avg_freq": staple_ratio,
        "is_diversified": 1.0 if crop_diversity >= 2 else 0.0,
        "primary_crop_freq": 0.5,
    }

def get_score(total_land, num_farms, crops):
    features = calculate_features(total_land, num_farms, crops)
    df = pd.DataFrame([features])
    result = scorer.predict(df)
    return {
        'score': int(result['idd_score'].iloc[0]),
        'risk_tier': result['risk_tier'].iloc[0],
        'features': features,
    }

# =============================================================================
# Main UI
# =============================================================================

# Header
col_title, col_spacer = st.columns([3, 1])
with col_title:
    st.markdown('<h1 style="font-size: 1.5rem; margin-bottom: 0;">IDD Credit Scoring Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; font-size: 0.875rem;">Agricultural Credit Assessment System</p>', unsafe_allow_html=True)

st.markdown("---")

# Input Section
with st.container():
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="section-header">Farm Details</div>', unsafe_allow_html=True)
        total_land = st.number_input(
            "Total land area",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.5,
            format="%.1f",
            help="Total agricultural land in hectares"
        )
        st.caption("Hectares")
        
        num_farms = st.number_input(
            "Number of Farm plots",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Number of separate land parcels"
        )
        st.caption("Separate land parcels")
    
    with col2:
        st.markdown('<div class="section-header">Crop Portfolio</div>', unsafe_allow_html=True)
        crop_options = ['Maize', 'Teff', 'Barley', 'Wheat', 'Onion', 'Pepper', 'Carrot', 'Cabbage', 'Rice', 'Bean', 'Pea', 'Coffee']
        selected_crops = st.multiselect(
            "Crops cultivated",
            options=crop_options,
            default=['Maize', 'Teff'],
            help="Select all crops grown by the farmer"
        )
        if selected_crops:
            st.caption(f"{len(selected_crops)} crop types selected")
        else:
            st.warning("Select at least one crop")

# Action Buttons
if selected_crops:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        show_features = st.button("View Features", use_container_width=True)
    with col_btn2:
        calculate_score = st.button("Calculate Score", use_container_width=True, type="primary")
else:
    show_features = False
    calculate_score = False

# =============================================================================
# Features Display
# =============================================================================
if show_features and selected_crops:
    features = calculate_features(total_land, num_farms, selected_crops)
    
    st.markdown("---")
    st.markdown('<div class="section-header">Feature Engineering Results</div>', unsafe_allow_html=True)
    
    # Feature grid
    cols = st.columns(4)
    feature_items = [
        ("num_farms", f"{features['num_farms']:.0f}", "Number of plots"),
        ("land_size_log", f"{features['land_size_log']:.4f}", "Log total area"),
        ("avg_land_log", f"{features['avg_land_log']:.4f}", "Log avg plot size"),
        ("crop_diversity", f"{features['crop_diversity']:.0f}", "Unique crop types"),
        ("multi_farm", f"{features['multi_farm']:.0f}", "Multiple plots flag"),
        ("staple_ratio", f"{features['staple_crop_avg_freq']:.3f}", "Staple crop proportion"),
        ("is_diversified", f"{features['is_diversified']:.0f}", "Diversification flag"),
        ("primary_freq", f"{features['primary_crop_freq']:.2f}", "Market frequency")
    ]
    
    for idx, (name, value, desc) in enumerate(feature_items):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{name}</div>
                <div style="font-size: 0.7rem; color: #9ca3af;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("Calculation Details"):
        avg_plot = total_land / num_farms
        staple_crops = [c for c in selected_crops if c in STAPLE_CROPS]
        st.markdown(f"""
        - **Average plot size:** {avg_plot:.2f} hectares
        - **Staple crops grown:** {', '.join(staple_crops) if staple_crops else 'None'}
        - **Crop diversity score:** {len(set(selected_crops))} unique types
        - **Multi-farm indicator:** {'Yes' if num_farms > 1 else 'No'}
        - **Diversified farmer:** {'Yes' if len(set(selected_crops)) >= 2 else 'No'}
        """)

# =============================================================================
# Score Calculation
# =============================================================================
if calculate_score and selected_crops:
    with st.spinner("Processing"):
        result = get_score(total_land, num_farms, selected_crops)
    
    st.markdown("---")
    
    # Score Display
    col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
    with col_score2:
        risk_class = "risk-low" if result['score'] >= 700 else "risk-medium" if result['score'] >= 550 else "risk-high"
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Credit Score</div>
            <div class="score-value">{result['score']}</div>
            <div class="risk-badge {risk_class}">{result['risk_tier']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Meter
    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
    risk_level = (result['score'] - 300) / 550
    st.progress(risk_level)
    
    if result['score'] >= 700:
        st.success("Credit Recommendation: Approved")
        st.caption("Low risk farmer with strong repayment capacity")
    elif result['score'] >= 550:
        st.warning("Credit Recommendation: Conditional Approval")
        st.caption("Medium risk - Consider additional collateral or shorter tenure")
    else:
        st.error("Credit Recommendation: Declined")
        st.caption("High risk - Insufficient creditworthiness")
    
    # Score Components
    st.markdown('<div class="section-header">Score Components</div>', unsafe_allow_html=True)
    
    features = calculate_features(total_land, num_farms, selected_crops)
    components = {
        "Farm Scale": min(30, (total_land / 100) * 30),
        "Land Quality": min(15, (total_land/num_farms / 10) * 15),
        "Spread": min(10, num_farms * 3.33),
        "Crop Diversity": min(15, (len(selected_crops) / 5) * 15),
        "Staple Ratio": min(10, features['staple_crop_avg_freq'] * 10),
        "Market Liquidity": 10
    }
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(components.keys()),
            y=list(components.values()),
            marker_color='#2c5a7a',
            text=[f"{v:.1f}" for v in components.values()],
            textposition='auto',
            textfont=dict(size=12)
        )
    ])
    
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="",
        yaxis_title="Points",
        yaxis_range=[0, 35],
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='#e5e7eb'),
        yaxis=dict(gridcolor='#e5e7eb')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Feature Details Toggle
    with st.expander("Technical Details"):
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown("**Clustering Features (8 dimensions)**")
            feature_df = pd.DataFrame([
                {"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else f"{v:.0f}"}
                for k, v in result['features'].items()
            ])
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        with col_f2:
            st.markdown("**Score Weights Applied**")
            weights_df = pd.DataFrame([
                {"Component": k.replace('_', ' ').title(), "Weight": f"{int(v*100)}%"}
                for k, v in scorer.SCORE_WEIGHTS.items()
            ])
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

elif calculate_score and not selected_crops:
    st.error("Selection Required")
    st.caption("Please select at least one crop type before calculating the credit score")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #9ca3af; font-size: 0.75rem; padding: 1rem 0;">
    IDD Credit Scoring Engine v{scorer.VERSION} | Powered by Agricultural Risk Analytics
</div>
""", unsafe_allow_html=True)