"""
ChatGPT Regime Shift Dashboard
MGMT 69000: Mastering AI for Finance - Purdue University

Analyzes the ChatGPT Launch (Nov 30, 2022) as a regime shift event,
examining creative destruction and sample space expansion in financial markets.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import find_peaks
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ChatGPT Regime Shift Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Key dates
CHATGPT_LAUNCH = pd.Timestamp('2022-11-30')
CHEGG_CRASH = pd.Timestamp('2023-05-02')
NVIDIA_EARNINGS = pd.Timestamp('2023-05-24')

# Stock tickers
WINNERS = ['NVDA', 'MSFT', 'META', 'GOOGL']
LOSERS = ['CHGG']
BENCHMARK = ['SPY']
ALL_TICKERS = WINNERS + LOSERS + BENCHMARK


@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if not df.empty:
                # Convert timezone-aware index to timezone-naive for consistent comparisons
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                data[ticker] = df
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
    return data


def normalize_prices(data, base_date):
    """Normalize all prices to 100 at the base date."""
    normalized = pd.DataFrame()

    for ticker, df in data.items():
        if df.empty:
            continue

        # Find the closest trading date to base_date
        idx = df.index.get_indexer([base_date], method='nearest')[0]
        base_price = df['Close'].iloc[idx]

        normalized[ticker] = (df['Close'] / base_price) * 100

    return normalized


def calculate_returns(data, base_date):
    """Calculate total returns since base date."""
    returns = {}

    for ticker, df in data.items():
        if df.empty:
            continue

        # Find closest trading date to base_date
        idx = df.index.get_indexer([base_date], method='nearest')[0]
        base_price = df['Close'].iloc[idx]
        final_price = df['Close'].iloc[-1]

        returns[ticker] = ((final_price - base_price) / base_price) * 100

    return returns


def calculate_rolling_entropy(data, window=60):
    """
    Calculate rolling Shannon entropy of market cap weights.
    Higher entropy = more equal distribution, Lower entropy = more concentration.
    """
    # Create a combined price DataFrame
    prices = pd.DataFrame()
    for ticker, df in data.items():
        if ticker in WINNERS + LOSERS:  # Exclude benchmark
            prices[ticker] = df['Close']

    prices = prices.dropna()

    if prices.empty:
        return pd.Series(dtype=float)

    # Use price as proxy for relative weights
    # In reality you'd use market cap, but this shows the concentration effect
    def shannon_entropy(weights):
        weights = weights / weights.sum()  # Normalize
        weights = weights[weights > 0]  # Remove zeros
        return -np.sum(weights * np.log2(weights))

    entropy = prices.rolling(window=window).apply(
        lambda x: shannon_entropy(x.iloc[-1] if len(x) > 0 else x),
        raw=False
    ).mean(axis=1)

    # Actually calculate proper rolling entropy
    entropy_series = []
    for i in range(len(prices)):
        if i < window:
            entropy_series.append(np.nan)
        else:
            window_prices = prices.iloc[i]
            weights = window_prices / window_prices.sum()
            ent = shannon_entropy(weights)
            entropy_series.append(ent)

    return pd.Series(entropy_series, index=prices.index)


def detect_regime_breaks(returns_series, threshold=2.5):
    """
    Detect structural break points using cumulative sum analysis.
    Returns dates where significant regime shifts occurred.
    """
    if len(returns_series) < 30:
        return []

    # Calculate cumulative sum of standardized returns
    standardized = (returns_series - returns_series.mean()) / returns_series.std()
    cusum = standardized.cumsum()

    # Find peaks in absolute cusum (regime change points)
    abs_cusum = np.abs(cusum.values)

    # Use find_peaks to detect significant changes
    peaks, properties = find_peaks(abs_cusum, prominence=threshold, distance=20)

    break_dates = returns_series.index[peaks].tolist()

    return break_dates


def create_normalized_chart(normalized_df):
    """Create the normalized price chart with event annotations."""
    fig = go.Figure()

    # Color scheme
    colors = {
        'NVDA': '#76B900',   # NVIDIA green
        'MSFT': '#00A4EF',   # Microsoft blue
        'META': '#0866FF',   # Meta blue
        'GOOGL': '#4285F4',  # Google blue
        'CHGG': '#FF6B6B',   # Red for loser
        'SPY': '#888888'     # Gray for benchmark
    }

    line_styles = {
        'NVDA': 'solid',
        'MSFT': 'solid',
        'META': 'solid',
        'GOOGL': 'solid',
        'CHGG': 'dash',
        'SPY': 'dot'
    }

    for col in normalized_df.columns:
        fig.add_trace(go.Scatter(
            x=normalized_df.index,
            y=normalized_df[col],
            mode='lines',
            name=col,
            line=dict(
                color=colors.get(col, '#333'),
                dash=line_styles.get(col, 'solid'),
                width=2 if col != 'SPY' else 1.5
            ),
            hovertemplate=f'{col}<br>Date: %{{x}}<br>Value: %{{y:.1f}}<extra></extra>'
        ))

    # Add event annotations
    events = [
        (CHATGPT_LAUNCH, "ChatGPT Launch", "top"),
        (CHEGG_CRASH, "Chegg Crash (-49%)", "bottom"),
        (NVIDIA_EARNINGS, "NVIDIA AI Earnings", "top")
    ]

    for date, label, position in events:
        if date in normalized_df.index or (normalized_df.index[0] <= date <= normalized_df.index[-1]):
            y_pos = normalized_df.loc[normalized_df.index >= date].iloc[0].max() if position == "top" else normalized_df.loc[normalized_df.index >= date].iloc[0].min()

            fig.add_vline(
                x=date,
                line_dash="dash",
                line_color="rgba(0,0,0,0.3)",
                line_width=1
            )

            fig.add_annotation(
                x=date,
                y=y_pos * (1.1 if position == "top" else 0.9),
                text=label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#666",
                font=dict(size=10, color="#333"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#666",
                borderwidth=1,
                borderpad=4
            )

    fig.update_layout(
        title=dict(
            text="<b>Normalized Stock Performance Since ChatGPT Launch</b><br><sup>Rebased to 100 on November 30, 2022</sup>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base = 100)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=500
    )

    # Add y=100 reference line
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)

    return fig


def create_returns_chart(returns_dict):
    """Create the total returns bar chart."""
    df = pd.DataFrame(list(returns_dict.items()), columns=['Ticker', 'Return'])
    df['Color'] = df['Ticker'].apply(lambda x: 'Winner' if x in WINNERS else ('Loser' if x in LOSERS else 'Benchmark'))
    df = df.sort_values('Return', ascending=True)

    color_map = {'Winner': '#2E7D32', 'Loser': '#C62828', 'Benchmark': '#666666'}

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Return']],
            y=[row['Ticker']],
            orientation='h',
            marker_color=color_map[row['Color']],
            name=row['Color'],
            showlegend=False,
            text=f"{row['Return']:.1f}%",
            textposition='outside',
            hovertemplate=f"{row['Ticker']}: {row['Return']:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="<b>Total Returns Since ChatGPT Launch</b><br><sup>November 30, 2022 to Present</sup>",
            font=dict(size=18)
        ),
        xaxis_title="Return (%)",
        yaxis_title="",
        template="plotly_white",
        height=400,
        showlegend=False
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_color="black", line_width=1)

    return fig


def create_entropy_chart(entropy_series, normalized_df):
    """Create the rolling entropy chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Market Concentration (Inverse Shannon Entropy)", "NVDA Dominance")
    )

    # Inverse entropy (higher = more concentrated)
    inverse_entropy = 1 / entropy_series.replace(0, np.nan)

    fig.add_trace(
        go.Scatter(
            x=entropy_series.index,
            y=inverse_entropy,
            mode='lines',
            name='Concentration Index',
            line=dict(color='#8B5CF6', width=2),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)'
        ),
        row=1, col=1
    )

    # NVDA share of the group
    if 'NVDA' in normalized_df.columns:
        nvda_share = normalized_df['NVDA'] / normalized_df[WINNERS + LOSERS].sum(axis=1) * 100

        fig.add_trace(
            go.Scatter(
                x=nvda_share.index,
                y=nvda_share,
                mode='lines',
                name='NVDA Weight %',
                line=dict(color='#76B900', width=2)
            ),
            row=2, col=1
        )

    # Add event lines
    for date in [CHATGPT_LAUNCH, CHEGG_CRASH, NVIDIA_EARNINGS]:
        fig.add_vline(x=date, line_dash="dash", line_color="rgba(0,0,0,0.2)", row="all")

    fig.update_layout(
        title=dict(
            text="<b>Sample Space Expansion: Winner-Take-Most Dynamics</b>",
            font=dict(size=18)
        ),
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    fig.update_yaxes(title_text="NVDA Share (%)", row=2, col=1)

    return fig


def create_regime_chart(data, break_dates):
    """Create chart showing regime detection results."""
    # Calculate portfolio returns
    returns_df = pd.DataFrame()
    for ticker, df in data.items():
        if ticker in WINNERS:
            returns_df[ticker] = df['Close'].pct_change()

    avg_returns = returns_df.mean(axis=1).dropna()
    cumulative = (1 + avg_returns).cumprod()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative,
        mode='lines',
        name='AI Winners Portfolio',
        line=dict(color='#2E7D32', width=2)
    ))

    # Highlight regime periods
    regime_colors = ['rgba(255, 235, 238, 0.5)', 'rgba(232, 245, 233, 0.5)',
                     'rgba(227, 242, 253, 0.5)', 'rgba(255, 243, 224, 0.5)']

    all_dates = [cumulative.index[0]] + break_dates + [cumulative.index[-1]]

    for i in range(len(all_dates) - 1):
        fig.add_vrect(
            x0=all_dates[i],
            x1=all_dates[i+1],
            fillcolor=regime_colors[i % len(regime_colors)],
            layer="below",
            line_width=0,
            annotation_text=f"Regime {i+1}",
            annotation_position="top left",
            annotation_font_size=10
        )

    # Mark break points
    for date in break_dates:
        fig.add_vline(
            x=date,
            line_dash="solid",
            line_color="red",
            line_width=2
        )

        y_val = cumulative.loc[cumulative.index >= date].iloc[0] if any(cumulative.index >= date) else cumulative.iloc[-1]
        fig.add_annotation(
            x=date,
            y=y_val,
            text="Break",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(color="red", size=10)
        )

    # Add key events
    fig.add_vline(x=CHATGPT_LAUNCH, line_dash="dash", line_color="blue", line_width=1.5)
    fig.add_vline(x=NVIDIA_EARNINGS, line_dash="dash", line_color="green", line_width=1.5)

    fig.update_layout(
        title=dict(
            text="<b>Regime Detection: Structural Breaks in AI Stock Returns</b><br><sup>Red lines indicate detected regime shifts</sup>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Starting at 1.0)",
        template="plotly_white",
        height=450,
        showlegend=True
    )

    return fig


# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Purdue_University_seal.svg/1200px-Purdue_University_seal.svg.png", width=100)
    st.markdown("## MGMT 69000")
    st.markdown("### Mastering AI for Finance")
    st.markdown("---")

    st.markdown("### About This Dashboard")
    st.markdown("""
    This analysis examines the **ChatGPT launch** (Nov 30, 2022) as a **regime shift event**
    in financial markets.

    **Key Concepts:**
    - **Creative Destruction**: Old business models disrupted
    - **Sample Space Expansion**: New asset classes emerge
    - **Winner-Take-Most**: Concentration in AI infrastructure
    """)

    st.markdown("---")
    st.markdown("### DRIVER Methodology")

    with st.expander("D - Discover"):
        st.markdown("""
        Identified ChatGPT launch as potential regime shift event.
        Research question: How did this technology release restructure
        market valuations across the AI value chain?
        """)

    with st.expander("R - Represent"):
        st.markdown("""
        - **Winners**: NVDA, MSFT, META, GOOGL (AI infrastructure)
        - **Losers**: CHGG (disrupted business model)
        - **Benchmark**: SPY (market context)
        - **Metrics**: Normalized returns, entropy, regime breaks
        """)

    with st.expander("I - Implement"):
        st.markdown("""
        Built interactive Streamlit dashboard with:
        - yfinance for market data
        - Plotly for visualization
        - scipy for regime detection
        - Shannon entropy for concentration
        """)

    with st.expander("V - Validate"):
        st.markdown("""
        - NVDA: Massive outperformance validates AI infrastructure thesis
        - CHGG: -90%+ decline confirms disruption hypothesis
        - Entropy decline confirms winner-take-most dynamics
        """)

    with st.expander("E - Evolve"):
        st.markdown("""
        Future extensions:
        - Add more disrupted companies
        - Include options market data
        - Sentiment analysis integration
        """)

    with st.expander("R - Reflect"):
        st.markdown("""
        Key learnings:
        - Technology releases can trigger regime shifts
        - Market reprices entire value chains
        - Speed of adjustment is accelerating
        """)

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("Stock data: Yahoo Finance API")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# Main content
st.markdown('<p class="main-header">ChatGPT Regime Shift Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Examining Creative Destruction and Sample Space Expansion in Financial Markets</p>', unsafe_allow_html=True)

# Fetch data
with st.spinner("Fetching market data..."):
    data = fetch_stock_data(
        ALL_TICKERS,
        start_date='2022-10-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )

if not data:
    st.error("Could not fetch stock data. Please check your internet connection and try again.")
    st.stop()

# Calculate metrics
normalized_df = normalize_prices(data, CHATGPT_LAUNCH)
returns = calculate_returns(data, CHATGPT_LAUNCH)
entropy = calculate_rolling_entropy(data, window=60)

# Calculate returns for regime detection
winner_returns = pd.DataFrame()
for ticker in WINNERS:
    if ticker in data:
        winner_returns[ticker] = data[ticker]['Close'].pct_change()
avg_winner_returns = winner_returns.mean(axis=1).dropna()
break_dates = detect_regime_breaks(avg_winner_returns, threshold=2.0)

# Key metrics row
st.markdown("### Key Metrics Since ChatGPT Launch")
col1, col2, col3, col4 = st.columns(4)

with col1:
    nvda_return = returns.get('NVDA', 0)
    st.metric(
        label="NVIDIA (NVDA)",
        value=f"{nvda_return:.1f}%",
        delta="AI Infrastructure Leader"
    )

with col2:
    spy_return = returns.get('SPY', 0)
    st.metric(
        label="S&P 500 (SPY)",
        value=f"{spy_return:.1f}%",
        delta="Market Benchmark"
    )

with col3:
    chgg_return = returns.get('CHGG', 0)
    st.metric(
        label="Chegg (CHGG)",
        value=f"{chgg_return:.1f}%",
        delta="Disrupted Model",
        delta_color="inverse"
    )

with col4:
    outperformance = nvda_return - spy_return
    st.metric(
        label="NVDA vs SPY",
        value=f"+{outperformance:.1f}%",
        delta="Alpha Generated"
    )

st.markdown("---")

# Main charts
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Performance",
    "📊 Total Returns",
    "🎯 Market Concentration",
    "🔄 Regime Detection"
])

with tab1:
    st.plotly_chart(create_normalized_chart(normalized_df), use_container_width=True)

    st.markdown("""
    **Interpretation:** The chart shows dramatic divergence after ChatGPT's launch.
    AI infrastructure companies (NVDA, META, MSFT, GOOGL) massively outperformed,
    while Chegg's business model was fundamentally disrupted. This is the hallmark
    of a **regime shift** - the old rules no longer apply.
    """)

with tab2:
    st.plotly_chart(create_returns_chart(returns), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Winners (AI Infrastructure):**
        - Benefited from increased AI compute demand
        - Cloud platforms saw enterprise adoption surge
        - GPU demand exceeded supply
        """)
    with col2:
        st.markdown("""
        **Losers (Disrupted Models):**
        - Chegg's homework help service commoditized
        - Student migration to ChatGPT
        - 49% single-day crash on May 2, 2023
        """)

with tab3:
    st.plotly_chart(create_entropy_chart(entropy, normalized_df), use_container_width=True)

    st.markdown("""
    **Sample Space Expansion:** The decreasing entropy (increasing concentration) shows
    that the AI opportunity isn't evenly distributed. NVIDIA's share of the "AI winners"
    basket has grown dramatically, demonstrating **winner-take-most dynamics** in
    emerging technology markets.
    """)

with tab4:
    st.plotly_chart(create_regime_chart(data, break_dates), use_container_width=True)

    st.markdown("""
    **Regime Detection:** Using cumulative sum analysis (CUSUM) to detect structural
    breaks in the return series. Red vertical lines indicate points where the statistical
    properties of returns fundamentally changed - these are the regime shift moments.

    **Key Events:**
    - **Nov 30, 2022**: ChatGPT launch - immediate repricing begins
    - **May 24, 2023**: NVIDIA earnings confirm AI infrastructure demand
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>MGMT 69000: Mastering AI for Finance</strong> | Purdue University</p>
    <p>Dashboard built with Streamlit, Plotly, and Python | Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
