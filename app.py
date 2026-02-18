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
from dotenv import load_dotenv
import os

load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ChatGPT Regime Shift Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

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

# Historical events for CUSUM validation
HISTORICAL_EVENTS = [
    ('iPhone Launch', pd.Timestamp('2007-01-09')),
    ('Bitcoin Surge', pd.Timestamp('2017-12-17')),
    ('COVID Crash', pd.Timestamp('2020-03-11')),
    ('ChatGPT Launch', pd.Timestamp('2022-11-30')),
]

AI_ANALYST_SYSTEM_PROMPT = """You are an AI Market Analyst embedded in the ChatGPT Regime Shift Dashboard, \
built for MGMT 69000: Mastering AI for Finance at Purdue University.

Context:
- This dashboard analyzes the ChatGPT launch (Nov 30, 2022) as a regime shift event in financial markets.
- It follows the DRIVER methodology (Discover, Represent, Implement, Validate, Evolve, Reflect).
- Tickers tracked — Winners (AI infrastructure): NVDA, MSFT, META, GOOGL. Losers (disrupted): CHGG. Benchmark: SPY.
- Core thesis: ChatGPT triggered creative destruction (old business models like Chegg disrupted) and sample space \
expansion (new AI-driven value chains emerged), producing winner-take-most dynamics.

Your role:
- Analyze the market data provided to you in the context of this regime shift thesis.
- Reference specific numbers, trends, and signals from the data.
- Connect observations to concepts like creative destruction, sample space expansion, entropy, and regime detection (CUSUM).
- Be concise but insightful — aim for 2-4 paragraphs per analysis.
- You may discuss market dynamics, statistical patterns, and historical parallels.
- NEVER provide financial advice, buy/sell recommendations, or price targets. Always frame analysis as educational.
- If asked about topics outside this dashboard's scope, politely redirect to the regime shift analysis."""


@st.cache_data(ttl=3600)
def fetch_risk_free_rate():
    """Fetch the 3-month Treasury yield from yfinance (^IRX = 13-week T-Bill).
    Returns the annualized rate as a percentage (e.g. 4.5 for 4.5%).
    Falls back to 4.5% if data is unavailable."""
    DEFAULT_RATE = 4.5
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            if pd.notna(rate) and 0 < rate < 20:
                return float(rate)
        return DEFAULT_RATE
    except Exception:
        return DEFAULT_RATE


@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data from Yahoo Finance, including shares outstanding for market cap."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if not df.empty:
                # Convert timezone-aware index to timezone-naive for consistent comparisons
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                # Fetch shares outstanding for market cap calculation
                try:
                    shares = stock.info.get('sharesOutstanding')
                    if shares and shares > 0:
                        df['SharesOutstanding'] = shares
                        df['MarketCap'] = df['Close'] * shares
                except Exception:
                    pass  # shares data unavailable; handled downstream
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
    Uses actual market cap (price x shares outstanding) when available,
    falls back to price as proxy otherwise.
    """
    # Build a DataFrame of market cap (or price as fallback) for each stock
    market_caps = pd.DataFrame()
    using_fallback = []
    for ticker, df in data.items():
        if ticker in WINNERS + LOSERS:  # Exclude benchmark
            if 'MarketCap' in df.columns:
                market_caps[ticker] = df['MarketCap']
            else:
                market_caps[ticker] = df['Close']
                using_fallback.append(ticker)

    market_caps = market_caps.dropna()

    if market_caps.empty:
        return pd.Series(dtype=float), []

    def shannon_entropy(weights):
        weights = weights / weights.sum()  # Normalize
        weights = weights[weights > 0]  # Remove zeros
        return -np.sum(weights * np.log2(weights))

    entropy_series = []
    for i in range(len(market_caps)):
        if i < window:
            entropy_series.append(np.nan)
        else:
            caps = market_caps.iloc[i]
            weights = caps / caps.sum()
            ent = shannon_entropy(weights)
            entropy_series.append(ent)

    return pd.Series(entropy_series, index=market_caps.index), using_fallback


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


def create_entropy_chart(entropy_series, data):
    """Create the rolling entropy chart using market cap weights."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Market Concentration (Inverse Shannon Entropy)", "NVDA Market Cap Share")
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

    # NVDA market cap share of the group
    market_caps = pd.DataFrame()
    for ticker, df in data.items():
        if ticker in WINNERS + LOSERS:
            if 'MarketCap' in df.columns:
                market_caps[ticker] = df['MarketCap']
            else:
                market_caps[ticker] = df['Close']
    market_caps = market_caps.dropna()

    if 'NVDA' in market_caps.columns and not market_caps.empty:
        nvda_share = market_caps['NVDA'] / market_caps.sum(axis=1) * 100

        fig.add_trace(
            go.Scatter(
                x=nvda_share.index,
                y=nvda_share,
                mode='lines',
                name='NVDA Market Cap %',
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
    fig.update_yaxes(title_text="NVDA Market Cap Share (%)", row=2, col=1)

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


def calculate_period_statistics(data, start_date, end_date, risk_free_rate=4.5):
    """Calculate statistics for a given period.
    risk_free_rate: annualized risk-free rate as a percentage (e.g. 4.5 for 4.5%)."""
    stats_data = []

    for ticker, df in data.items():
        # Filter to period
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_df = df.loc[mask]

        if len(period_df) < 2:
            continue

        daily_returns = period_df['Close'].pct_change().dropna()

        if len(daily_returns) < 2:
            continue

        avg_return = daily_returns.mean() * 252 * 100  # Annualized %
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized %
        sharpe = ((avg_return - risk_free_rate) / volatility) if volatility > 0 else 0

        stats_data.append({
            'Ticker': ticker,
            'Avg Daily Return (Ann. %)': avg_return,
            'Volatility (Ann. %)': volatility,
            'Sharpe Ratio': sharpe
        })

    return pd.DataFrame(stats_data)


def create_comparison_table(data, base_date, risk_free_rate=4.5):
    """Create pre vs post ChatGPT statistics comparison table."""
    # 6 months before and after
    pre_start = base_date - pd.Timedelta(days=180)
    pre_end = base_date - pd.Timedelta(days=1)
    post_start = base_date
    post_end = base_date + pd.Timedelta(days=180)

    pre_stats = calculate_period_statistics(data, pre_start, pre_end, risk_free_rate)
    post_stats = calculate_period_statistics(data, post_start, post_end, risk_free_rate)

    if pre_stats.empty or post_stats.empty:
        return None

    # Merge and format
    pre_stats = pre_stats.set_index('Ticker').add_suffix(' (Pre)')
    post_stats = post_stats.set_index('Ticker').add_suffix(' (Post)')

    combined = pre_stats.join(post_stats, how='outer')

    # Reorder columns for better comparison
    cols = []
    for metric in ['Avg Daily Return (Ann. %)', 'Volatility (Ann. %)', 'Sharpe Ratio']:
        cols.extend([f'{metric} (Pre)', f'{metric} (Post)'])

    combined = combined[[c for c in cols if c in combined.columns]]

    return combined.reset_index()


def create_correlation_heatmaps(data, base_date):
    """Create side-by-side correlation heatmaps for pre and post ChatGPT."""
    # 6 months before and after
    pre_start = base_date - pd.Timedelta(days=180)
    pre_end = base_date - pd.Timedelta(days=1)
    post_start = base_date
    post_end = base_date + pd.Timedelta(days=180)

    # Build returns DataFrames
    def get_returns_df(start, end):
        returns = pd.DataFrame()
        for ticker, df in data.items():
            mask = (df.index >= start) & (df.index <= end)
            period_df = df.loc[mask]
            if not period_df.empty:
                returns[ticker] = period_df['Close'].pct_change()
        return returns.dropna()

    pre_returns = get_returns_df(pre_start, pre_end)
    post_returns = get_returns_df(post_start, post_end)

    if pre_returns.empty or post_returns.empty:
        return None

    pre_corr = pre_returns.corr()
    post_corr = post_returns.corr()

    # Create side-by-side heatmaps
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Pre-ChatGPT Correlations<br>(Jun - Nov 2022)",
            "Post-ChatGPT Correlations<br>(Dec 2022 - May 2023)"
        ),
        horizontal_spacing=0.15
    )

    # Color scale
    colorscale = 'RdBu_r'

    # Pre-ChatGPT heatmap
    fig.add_trace(
        go.Heatmap(
            z=pre_corr.values,
            x=pre_corr.columns,
            y=pre_corr.index,
            colorscale=colorscale,
            zmin=-1, zmax=1,
            text=np.round(pre_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=False,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Post-ChatGPT heatmap
    fig.add_trace(
        go.Heatmap(
            z=post_corr.values,
            x=post_corr.columns,
            y=post_corr.index,
            colorscale=colorscale,
            zmin=-1, zmax=1,
            text=np.round(post_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title='Correlation', x=1.02),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text="<b>Correlation Structure: Before vs After ChatGPT Launch</b>",
            font=dict(size=18)
        ),
        template="plotly_white",
        height=450,
    )

    return fig


def create_chegg_crash_chart(data):
    """Create cumulative return chart highlighting the Chegg crash period."""
    # Calculate cumulative returns for all stocks
    cumulative = pd.DataFrame()

    for ticker, df in data.items():
        returns = df['Close'].pct_change()
        cumulative[ticker] = (1 + returns).cumprod()

    cumulative = cumulative.dropna()

    if cumulative.empty:
        return None

    # Normalize to start at 1
    cumulative = cumulative / cumulative.iloc[0]

    fig = go.Figure()

    # Color scheme
    colors = {
        'NVDA': '#76B900',
        'MSFT': '#00A4EF',
        'META': '#0866FF',
        'GOOGL': '#4285F4',
        'CHGG': '#FF6B6B',
        'SPY': '#888888'
    }

    # Add traces for each stock
    for col in cumulative.columns:
        line_width = 3 if col == 'CHGG' else 1.5
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col],
            mode='lines',
            name=col,
            line=dict(
                color=colors.get(col, '#333'),
                width=line_width
            ),
            hovertemplate=f'{col}<br>Date: %{{x}}<br>Cumulative Return: %{{y:.2f}}x<extra></extra>'
        ))

    # Highlight Chegg crash period (April 25 - May 10, 2023)
    crash_start = pd.Timestamp('2023-04-25')
    crash_end = pd.Timestamp('2023-05-10')

    fig.add_vrect(
        x0=crash_start,
        x1=crash_end,
        fillcolor="rgba(255, 0, 0, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Chegg Crash Zone",
        annotation_position="top left",
        annotation_font=dict(size=11, color="red")
    )

    # Add specific crash date annotation
    fig.add_vline(
        x=CHEGG_CRASH,
        line_dash="solid",
        line_color="red",
        line_width=2
    )

    fig.add_annotation(
        x=CHEGG_CRASH,
        y=0.3,
        text="May 2, 2023<br>Chegg -49%<br>in single day",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        font=dict(size=10, color="red"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="red",
        borderwidth=1
    )

    # Add ChatGPT launch line
    fig.add_vline(
        x=CHATGPT_LAUNCH,
        line_dash="dash",
        line_color="blue",
        line_width=1.5
    )

    fig.add_annotation(
        x=CHATGPT_LAUNCH,
        y=cumulative.max().max() * 0.9,
        text="ChatGPT<br>Launch",
        showarrow=False,
        font=dict(size=10, color="blue"),
        bgcolor="rgba(255,255,255,0.8)"
    )

    fig.update_layout(
        title=dict(
            text="<b>Cumulative Returns: The Chegg Crash in Context</b><br><sup>Highlighting the creative destruction moment</sup>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Starting at 1.0x)",
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    # Add horizontal line at 1.0
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)

    return fig


def simulate_portfolio(data, start_date, initial_investment=10000):
    """
    Simulate portfolio performance for three strategies:
    1. AI Winners: Equal weight NVDA, MSFT, META, GOOGL
    2. Benchmark: 100% SPY
    3. Disrupted: 100% CHGG
    """
    portfolios = {}

    # Get common date range starting from start_date
    all_dates = None
    for ticker, df in data.items():
        mask = df.index >= start_date
        dates = df.index[mask]
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(set(dates))

    if not all_dates:
        return None, None

    all_dates = sorted(list(all_dates))

    # Strategy 1: AI Winners (equal weight)
    winners_available = [t for t in WINNERS if t in data]
    if winners_available:
        per_stock = initial_investment / len(winners_available)
        portfolio_value = pd.Series(index=all_dates, dtype=float)

        for date in all_dates:
            total = 0
            for ticker in winners_available:
                df = data[ticker]
                if date in df.index:
                    # Find start price
                    start_idx = df.index.get_indexer([start_date], method='nearest')[0]
                    start_price = df['Close'].iloc[start_idx]
                    current_price = df.loc[date, 'Close']
                    shares = per_stock / start_price
                    total += shares * current_price
            portfolio_value[date] = total

        portfolios['AI Winners'] = portfolio_value

    # Strategy 2: SPY Benchmark
    if 'SPY' in data:
        df = data['SPY']
        start_idx = df.index.get_indexer([start_date], method='nearest')[0]
        start_price = df['Close'].iloc[start_idx]
        shares = initial_investment / start_price

        portfolio_value = pd.Series(index=all_dates, dtype=float)
        for date in all_dates:
            if date in df.index:
                portfolio_value[date] = shares * df.loc[date, 'Close']

        portfolios['SPY Benchmark'] = portfolio_value

    # Strategy 3: CHGG (Disrupted)
    if 'CHGG' in data:
        df = data['CHGG']
        start_idx = df.index.get_indexer([start_date], method='nearest')[0]
        start_price = df['Close'].iloc[start_idx]
        shares = initial_investment / start_price

        portfolio_value = pd.Series(index=all_dates, dtype=float)
        for date in all_dates:
            if date in df.index:
                portfolio_value[date] = shares * df.loc[date, 'Close']

        portfolios['CHGG (Disrupted)'] = portfolio_value

    # Create summary
    summary_data = []
    for name, values in portfolios.items():
        if len(values) > 0:
            final_value = values.iloc[-1]
            total_return = ((final_value - initial_investment) / initial_investment) * 100
            summary_data.append({
                'Strategy': name,
                'Initial Investment': f'${initial_investment:,.0f}',
                'Final Value': f'${final_value:,.0f}',
                'Total Return': f'{total_return:+.1f}%',
                'Return_numeric': total_return
            })

    summary_df = pd.DataFrame(summary_data)

    return portfolios, summary_df


@st.cache_data(ttl=3600)
def fetch_historical_event_data():
    """Fetch SPY data around each historical event for CUSUM validation."""
    event_data = {}
    for name, date in HISTORICAL_EVENTS:
        start = (date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        end = (date + pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        try:
            spy = yf.Ticker('SPY')
            df = spy.history(start=start, end=end)
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                event_data[name] = df
        except Exception:
            pass
    return event_data


def validate_cusum_on_event(returns_series, event_date, threshold=2.0):
    """Validate CUSUM detector against a known historical event.
    Returns (detected, days_delta, closest_break_date)."""
    break_dates = detect_regime_breaks(returns_series, threshold=threshold)
    if not break_dates:
        return False, None, None

    # Find closest break to the actual event
    deltas = [(bd, (bd - event_date).days) for bd in break_dates]
    closest = min(deltas, key=lambda x: abs(x[1]))
    return True, closest[1], closest[0]


def calculate_regime_probability(recent_returns, full_returns):
    """Calculate regime shift probability using CUSUM percentile rank.
    recent_returns: last ~6 months of winner portfolio returns
    full_returns: full history of winner portfolio returns for baseline distribution
    Returns (probability, cusum_series, current_value)."""
    if len(recent_returns) < 30 or len(full_returns) < 30:
        return 0.0, pd.Series(dtype=float), 0.0

    # Compute CUSUM on the full history for distribution baseline
    full_std = full_returns.std()
    full_mean = full_returns.mean()
    if full_std == 0:
        return 0.0, pd.Series(dtype=float), 0.0

    full_standardized = (full_returns - full_mean) / full_std
    full_cusum = full_standardized.cumsum()
    full_abs_cusum = np.abs(full_cusum)

    # Compute CUSUM on recent window
    recent_standardized = (recent_returns - full_mean) / full_std
    recent_cusum = recent_standardized.cumsum()
    current_value = float(np.abs(recent_cusum.iloc[-1]))

    # Percentile rank of current magnitude against full history
    probability = float(stats.percentileofscore(full_abs_cusum.dropna(), current_value))

    return probability, recent_cusum, current_value


def assess_early_warnings(data, entropy_series):
    """Assess early warning signals for regime shift.
    Returns dict with entropy_trend, correlation_shift, volatility_regime."""
    warnings_out = {}

    # 1. Entropy trend: slope of last 60 days
    if len(entropy_series.dropna()) >= 60:
        recent_entropy = entropy_series.dropna().iloc[-60:]
        x = np.arange(len(recent_entropy))
        slope, _, _, _, _ = stats.linregress(x, recent_entropy.values)
        # Falling entropy = increasing concentration = warning
        if slope < -0.005:
            status = 'red'
            direction = 'Falling (concentration increasing)'
        elif slope < 0:
            status = 'yellow'
            direction = 'Slightly falling'
        else:
            status = 'green'
            direction = 'Stable or rising'
        warnings_out['entropy_trend'] = {
            'value': float(slope),
            'direction': direction,
            'status': status,
            'label': 'Entropy Trend'
        }
    else:
        warnings_out['entropy_trend'] = {
            'value': 0.0, 'direction': 'Insufficient data',
            'status': 'yellow', 'label': 'Entropy Trend'
        }

    # 2. Correlation shift: last 30d avg pairwise corr vs 90d-ago baseline
    winner_returns = pd.DataFrame()
    for ticker in WINNERS:
        if ticker in data:
            winner_returns[ticker] = data[ticker]['Close'].pct_change()
    winner_returns = winner_returns.dropna()

    if len(winner_returns) >= 120:
        recent_corr = winner_returns.iloc[-30:].corr()
        baseline_corr = winner_returns.iloc[-120:-90].corr()
        # Average off-diagonal correlations
        mask = np.ones(recent_corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        recent_avg = recent_corr.values[mask].mean()
        baseline_avg = baseline_corr.values[mask].mean()
        corr_change = recent_avg - baseline_avg

        if corr_change > 0.15:
            status = 'red'
            direction = f'Rising (+{corr_change:.2f}) — herding signal'
        elif corr_change > 0.05:
            status = 'yellow'
            direction = f'Slightly rising (+{corr_change:.2f})'
        else:
            status = 'green'
            direction = f'Stable ({corr_change:+.2f})'
        warnings_out['correlation_shift'] = {
            'value': float(corr_change),
            'direction': direction,
            'status': status,
            'label': 'Correlation Structure'
        }
    else:
        warnings_out['correlation_shift'] = {
            'value': 0.0, 'direction': 'Insufficient data',
            'status': 'yellow', 'label': 'Correlation Structure'
        }

    # 3. Volatility regime: last 30d realized vol vs 90d average
    if len(winner_returns) >= 90:
        avg_returns = winner_returns.mean(axis=1)
        recent_vol = avg_returns.iloc[-30:].std() * np.sqrt(252)
        baseline_vol = avg_returns.iloc[-90:].std() * np.sqrt(252)
        vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

        if vol_ratio > 1.5:
            status = 'red'
            direction = f'Elevated ({vol_ratio:.1f}x baseline)'
        elif vol_ratio > 1.2:
            status = 'yellow'
            direction = f'Slightly elevated ({vol_ratio:.1f}x baseline)'
        else:
            status = 'green'
            direction = f'Normal ({vol_ratio:.1f}x baseline)'
        warnings_out['volatility_regime'] = {
            'value': float(vol_ratio),
            'direction': direction,
            'status': status,
            'label': 'Volatility Regime'
        }
    else:
        warnings_out['volatility_regime'] = {
            'value': 1.0, 'direction': 'Insufficient data',
            'status': 'yellow', 'label': 'Volatility Regime'
        }

    return warnings_out


def gather_market_context(data, returns, entropy, break_dates, avg_winner_returns, risk_free_rate):
    """Collect current dashboard data into a formatted string for the AI analyst."""
    lines = []

    # Sorted returns
    sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    lines.append("=== Current Returns Since ChatGPT Launch ===")
    for ticker, ret in sorted_returns:
        lines.append(f"  {ticker}: {ret:+.1f}%")
    if sorted_returns:
        lines.append(f"  Top performer: {sorted_returns[0][0]} ({sorted_returns[0][1]:+.1f}%)")
        lines.append(f"  Worst performer: {sorted_returns[-1][0]} ({sorted_returns[-1][1]:+.1f}%)")

    # Regime probability
    if len(avg_winner_returns) >= 126:
        recent = avg_winner_returns.iloc[-126:]
    else:
        recent = avg_winner_returns
    probability, _, current_cusum = calculate_regime_probability(recent, avg_winner_returns)
    lines.append(f"\n=== Regime Status ===")
    lines.append(f"  Regime shift probability (CUSUM percentile): {probability:.0f}%")
    lines.append(f"  Current CUSUM magnitude: {current_cusum:.2f}")

    # Early warning signals
    warning_signals = assess_early_warnings(data, entropy)
    lines.append(f"\n=== Early Warning Signals ===")
    for key, signal in warning_signals.items():
        lines.append(f"  {signal['label']}: {signal['direction']} (status: {signal['status']})")

    # Entropy
    clean_entropy = entropy.dropna()
    if len(clean_entropy) > 0:
        lines.append(f"\n=== Market Concentration (Shannon Entropy) ===")
        lines.append(f"  Latest entropy: {clean_entropy.iloc[-1]:.4f}")
        if len(clean_entropy) >= 30:
            lines.append(f"  30-day average: {clean_entropy.iloc[-30:].mean():.4f}")

    # Break dates
    if break_dates:
        lines.append(f"\n=== Detected Regime Breaks ===")
        for bd in break_dates:
            lines.append(f"  {bd.strftime('%Y-%m-%d')}")

    lines.append(f"\n=== Other ===")
    lines.append(f"  Risk-free rate (3M T-Bill): {risk_free_rate:.2f}%")
    lines.append(f"  Analysis date: {datetime.now().strftime('%Y-%m-%d')}")

    return "\n".join(lines)


def call_openai_analyst(messages, api_key):
    """Call OpenRouter API for market analysis. Returns (success: bool, text: str)."""
    if not OPENAI_AVAILABLE:
        return False, "The `openai` package is not installed. Run `pip install openai` to enable AI analysis."

    if not api_key:
        return False, "No API key found. Set `OPENAI_API_KEY` in your `.env` file to enable AI analysis."

    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/oyu-umveisharma/chatgpt-regime-shift",
                "X-Title": "ChatGPT Regime Shift Dashboard",
            },
        )
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": AI_ANALYST_SYSTEM_PROMPT},
            ] + messages,
        )
        return True, response.choices[0].message.content
    except openai.AuthenticationError:
        return False, "Invalid API key. Please check your `OPENAI_API_KEY` in the `.env` file."
    except openai.RateLimitError:
        return False, "Rate limit exceeded. Please wait a moment and try again."
    except openai.APIError as e:
        return False, f"OpenRouter API error: {e}"
    except Exception as e:
        return False, f"Unexpected error calling OpenRouter: {e}"


def create_portfolio_chart(portfolios, start_date):
    """Create portfolio value over time chart."""
    if not portfolios:
        return None

    fig = go.Figure()

    colors = {
        'AI Winners': '#2E7D32',
        'SPY Benchmark': '#666666',
        'CHGG (Disrupted)': '#C62828'
    }

    line_styles = {
        'AI Winners': 'solid',
        'SPY Benchmark': 'dot',
        'CHGG (Disrupted)': 'dash'
    }

    for name, values in portfolios.items():
        fig.add_trace(go.Scatter(
            x=values.index,
            y=values.values,
            mode='lines',
            name=name,
            line=dict(
                color=colors.get(name, '#333'),
                dash=line_styles.get(name, 'solid'),
                width=2.5
            ),
            hovertemplate=f'{name}<br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))

    # Add initial investment reference line
    fig.add_hline(y=10000, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Initial $10,000", annotation_position="right")

    fig.update_layout(
        title=dict(
            text=f"<b>Portfolio Simulator: $10,000 Investment</b><br><sup>Starting {start_date.strftime('%B %d, %Y')}</sup>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        yaxis=dict(tickformat='$,.0f')
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
    risk_free_rate = fetch_risk_free_rate()

if not data:
    st.error("Could not fetch stock data. Please check your internet connection and try again.")
    st.stop()

# Calculate metrics
normalized_df = normalize_prices(data, CHATGPT_LAUNCH)
returns = calculate_returns(data, CHATGPT_LAUNCH)
entropy, entropy_fallback_tickers = calculate_rolling_entropy(data, window=60)

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Price Performance",
    "📊 Total Returns",
    "🎯 Market Concentration",
    "🔄 Regime Detection",
    "🔬 Deep Analysis",
    "💰 Portfolio Impact",
    "🔮 Regime Prediction",
    "🤖 AI Market Analyst"
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
    st.plotly_chart(create_entropy_chart(entropy, data), use_container_width=True)

    if entropy_fallback_tickers:
        st.warning(f"Shares outstanding data unavailable for **{', '.join(entropy_fallback_tickers)}**. "
                   "Using stock price as a proxy for those tickers.")

    st.markdown("""
    **Sample Space Expansion:** The decreasing entropy (increasing concentration) shows
    that the AI opportunity isn't evenly distributed. NVIDIA's share of the "AI winners"
    basket has grown dramatically, demonstrating **winner-take-most dynamics** in
    emerging technology markets.

    *Methodology: Shannon entropy is calculated using actual market capitalization weights
    (price x shares outstanding) rather than raw stock prices, giving a true picture of
    economic concentration across these companies.*
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

with tab5:
    st.markdown("### Pre vs Post ChatGPT: Statistical Comparison")
    st.markdown("*Comparing 6 months before (Jun-Nov 2022) vs 6 months after (Dec 2022-May 2023)*")

    st.info(f"**Risk-Free Rate:** {risk_free_rate:.2f}% annualized (3-Month Treasury Bill, ^IRX). "
            "Sharpe ratios are calculated as (annualized return - risk-free rate) / annualized volatility.")

    # Statistics comparison table
    comparison_table = create_comparison_table(data, CHATGPT_LAUNCH, risk_free_rate)
    if comparison_table is not None:
        # Style the dataframe
        def style_comparison(val, metric):
            if pd.isna(val):
                return ''
            if 'Return' in metric:
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'
            return ''

        st.dataframe(
            comparison_table.style.format({
                col: '{:.2f}' for col in comparison_table.columns if col != 'Ticker'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        **Key Observations:**
        - **NVDA**: Sharpe ratio dramatically improved post-ChatGPT, reflecting the AI infrastructure thesis
        - **CHGG**: Returns turned deeply negative with elevated volatility - classic disruption signature
        - **Volatility changes**: Winners saw volatility increase (opportunity), losers saw volatility spike (risk)
        """)
    else:
        st.warning("Insufficient data for statistical comparison")

    st.markdown("---")

    # Correlation heatmaps
    st.markdown("### Correlation Structure Shift")

    corr_fig = create_correlation_heatmaps(data, CHATGPT_LAUNCH)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - **Pre-ChatGPT**: Tech stocks moved largely together with the market
        - **Post-ChatGPT**: AI winners (NVDA, MSFT, META, GOOGL) decoupled from the benchmark
        - **CHGG correlation shift**: Chegg's correlation to tech leaders weakened as it became a disruption story
        """)
    else:
        st.warning("Insufficient data for correlation analysis")

    st.markdown("---")

    # Chegg crash chart
    st.markdown("### The Chegg Crash: Creative Destruction in Action")

    crash_fig = create_chegg_crash_chart(data)
    if crash_fig is not None:
        st.plotly_chart(crash_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What Happened on May 2, 2023:**
            - Chegg reported Q1 2023 earnings
            - CEO Dan Rosensweig cited ChatGPT impact
            - Stock dropped **49% in a single day**
            - Largest single-day drop in company history
            """)
        with col2:
            st.markdown("""
            **The Creative Destruction Thesis:**
            - ChatGPT commoditized homework help
            - Students migrated to free AI tools
            - Chegg's subscription model disrupted
            - Classic "sample space contraction" for incumbents
            """)
    else:
        st.warning("Insufficient data for crash analysis")

with tab6:
    st.markdown("### Portfolio Impact Simulator")
    st.markdown("*Compare investment strategies: AI Winners vs Market Benchmark vs Disrupted Stock*")

    # Date slider for "What If" analysis
    st.markdown("#### What If: Adjust Your Start Date")

    # Get the date range from the data
    min_date = pd.Timestamp('2022-10-01')
    max_date = pd.Timestamp('2024-01-01')

    # Create date slider
    selected_date = st.slider(
        "Select investment start date:",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=CHATGPT_LAUNCH.to_pydatetime(),
        format="MMM DD, YYYY",
        help="Drag to see how results change based on when you invested"
    )

    selected_timestamp = pd.Timestamp(selected_date)

    # Quick date buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("ChatGPT Launch (Nov 30, 2022)", use_container_width=True):
            selected_timestamp = CHATGPT_LAUNCH
    with col2:
        if st.button("Start of 2023", use_container_width=True):
            selected_timestamp = pd.Timestamp('2023-01-03')
    with col3:
        if st.button("Before Chegg Crash (Apr 2023)", use_container_width=True):
            selected_timestamp = pd.Timestamp('2023-04-01')
    with col4:
        if st.button("Mid 2023", use_container_width=True):
            selected_timestamp = pd.Timestamp('2023-07-01')

    st.markdown("---")

    # Run simulation
    portfolios, summary_df = simulate_portfolio(data, selected_timestamp, initial_investment=10000)

    if portfolios and summary_df is not None and not summary_df.empty:
        # Portfolio chart
        portfolio_fig = create_portfolio_chart(portfolios, selected_timestamp)
        if portfolio_fig:
            st.plotly_chart(portfolio_fig, use_container_width=True)

        # Summary table
        st.markdown("#### Investment Summary")

        # Style the summary table
        display_df = summary_df[['Strategy', 'Initial Investment', 'Final Value', 'Total Return']].copy()

        # Color code based on return
        def color_return(val):
            if '+' in str(val):
                return 'color: green; font-weight: bold'
            elif '-' in str(val):
                return 'color: red; font-weight: bold'
            return ''

        styled_df = display_df.style.applymap(color_return, subset=['Total Return'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Key insights
        st.markdown("---")
        st.markdown("#### Key Insights")

        if len(summary_df) >= 3:
            ai_return = summary_df[summary_df['Strategy'] == 'AI Winners']['Return_numeric'].values
            spy_return = summary_df[summary_df['Strategy'] == 'SPY Benchmark']['Return_numeric'].values
            chgg_return = summary_df[summary_df['Strategy'] == 'CHGG (Disrupted)']['Return_numeric'].values

            col1, col2, col3 = st.columns(3)

            with col1:
                if len(ai_return) > 0 and len(spy_return) > 0:
                    alpha = ai_return[0] - spy_return[0]
                    st.metric(
                        "AI Winners Alpha vs SPY",
                        f"{alpha:+.1f}%",
                        delta="Outperformance" if alpha > 0 else "Underperformance"
                    )

            with col2:
                if len(ai_return) > 0 and len(chgg_return) > 0:
                    spread = ai_return[0] - chgg_return[0]
                    st.metric(
                        "Winner vs Loser Spread",
                        f"{spread:+.1f}%",
                        delta="Creative Destruction Gap"
                    )

            with col3:
                if len(chgg_return) > 0:
                    st.metric(
                        "Disruption Impact (CHGG)",
                        f"{chgg_return[0]:+.1f}%",
                        delta="Value Destroyed",
                        delta_color="inverse"
                    )

        st.markdown("""
        **Investment Lessons from the AI Regime Shift:**
        - **Timing matters, but direction matters more**: Even investing after the initial surge, AI infrastructure outperformed
        - **Disruption is permanent**: CHGG never recovered - this wasn't a dip to buy
        - **Concentration risk**: The AI winners diverged significantly from the broad market (SPY)
        - **Regime shifts create asymmetric outcomes**: Winners gained multiples of what the market returned; losers lost most of their value
        """)

    else:
        st.warning("Insufficient data for portfolio simulation. Try selecting a different date range.")

with tab7:
    st.markdown("### Regime Prediction & Early Warning System")
    st.markdown("*Validating the CUSUM detector against history and surfacing forward-looking signals*")

    # --- Section A: Historical Validation ---
    st.markdown("#### Historical Validation")
    st.markdown("Testing whether our CUSUM regime detector would have flagged known market-moving events using SPY returns.")

    with st.spinner("Fetching historical event data..."):
        event_data = fetch_historical_event_data()

    if event_data:
        val_cols = st.columns(len(HISTORICAL_EVENTS))
        for i, (event_name, event_date) in enumerate(HISTORICAL_EVENTS):
            with val_cols[i]:
                if event_name in event_data:
                    spy_returns = event_data[event_name]['Close'].pct_change().dropna()
                    detected, days_delta, closest_date = validate_cusum_on_event(
                        spy_returns, event_date, threshold=2.0
                    )
                    if detected:
                        delta_label = f"{days_delta:+d} days" if days_delta is not None else "N/A"
                        st.metric(
                            label=event_name,
                            value="Detected",
                            delta=delta_label
                        )
                        st.caption(f"Event: {event_date.strftime('%Y-%m-%d')}")
                        if closest_date:
                            st.caption(f"Break: {closest_date.strftime('%Y-%m-%d')}")
                    else:
                        st.metric(
                            label=event_name,
                            value="Not Detected",
                            delta="No break found",
                            delta_color="off"
                        )
                        st.caption(f"Event: {event_date.strftime('%Y-%m-%d')}")
                else:
                    st.metric(
                        label=event_name,
                        value="No Data",
                        delta="Fetch failed",
                        delta_color="off"
                    )
    else:
        st.warning("Could not fetch historical event data for validation.")

    st.markdown("---")

    # --- Section B: Current Regime Status ---
    st.markdown("#### Current Regime Status")

    # Use last 6 months of winner returns for probability
    if len(avg_winner_returns) >= 126:
        recent_returns = avg_winner_returns.iloc[-126:]
    else:
        recent_returns = avg_winner_returns

    probability, cusum_series, current_cusum = calculate_regime_probability(
        recent_returns, avg_winner_returns
    )

    prob_col1, prob_col2 = st.columns([1, 2])

    with prob_col1:
        st.metric(
            label="Regime Shift Probability",
            value=f"{probability:.0f}%",
            delta="Based on CUSUM percentile rank"
        )

        # Gauge chart
        if probability < 30:
            gauge_color = "green"
        elif probability < 60:
            gauge_color = "orange"
        else:
            gauge_color = "red"

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0,200,0,0.15)'},
                    {'range': [30, 60], 'color': 'rgba(255,165,0,0.15)'},
                    {'range': [60, 100], 'color': 'rgba(255,0,0,0.15)'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 2},
                    'thickness': 0.75,
                    'value': probability
                }
            },
            title={'text': 'Shift Probability'}
        ))
        gauge_fig.update_layout(height=250, margin=dict(t=40, b=0, l=30, r=30))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with prob_col2:
        # CUSUM line chart with threshold bands
        if len(cusum_series) > 0:
            cusum_fig = go.Figure()
            cusum_fig.add_trace(go.Scatter(
                x=cusum_series.index,
                y=cusum_series.values,
                mode='lines',
                name='CUSUM',
                line=dict(color='#1E3A5F', width=2)
            ))
            # Add threshold bands
            cusum_fig.add_hline(y=2.0, line_dash="dash", line_color="orange",
                                annotation_text="Warning threshold", annotation_position="top left")
            cusum_fig.add_hline(y=-2.0, line_dash="dash", line_color="orange")
            cusum_fig.add_hline(y=3.0, line_dash="dash", line_color="red",
                                annotation_text="Critical threshold", annotation_position="top left")
            cusum_fig.add_hline(y=-3.0, line_dash="dash", line_color="red")
            cusum_fig.update_layout(
                title="Recent CUSUM Values (Last 6 Months)",
                xaxis_title="Date",
                yaxis_title="CUSUM Value",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(cusum_fig, use_container_width=True)
        else:
            st.info("Insufficient data to compute CUSUM series.")

    st.markdown("---")

    # --- Section C: Early Warning Signals ---
    st.markdown("#### Early Warning Signals")

    warning_signals = assess_early_warnings(data, entropy)

    signal_cols = st.columns(3)
    for idx, (key, signal) in enumerate(warning_signals.items()):
        with signal_cols[idx]:
            color_map = {'green': '#28a745', 'yellow': '#ffc107', 'red': '#dc3545'}
            bg_map = {'green': '#d4edda', 'yellow': '#fff3cd', 'red': '#f8d7da'}
            color = color_map.get(signal['status'], '#6c757d')
            bg = bg_map.get(signal['status'], '#e2e3e5')

            st.markdown(
                f"""<div style="background-color: {bg}; border-left: 5px solid {color};
                padding: 1rem; border-radius: 5px; margin-bottom: 0.5rem;">
                <strong style="color: {color};">{signal['label']}</strong><br>
                <span style="font-size: 0.95rem;">{signal['direction']}</span>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("""
    **Signal Explanations:**
    - **Entropy Trend**: Measures whether market concentration is increasing. Falling entropy suggests a winner-take-most dynamic is intensifying.
    - **Correlation Structure**: Compares recent pairwise correlation among AI winners to their 90-day baseline. Rising correlation signals herding behavior.
    - **Volatility Regime**: Compares recent 30-day realized volatility to the 90-day average. Elevated volatility often precedes or accompanies regime shifts.
    """)

    st.markdown("---")

    # --- Section D: Disclaimer ---
    st.warning(
        "**Disclaimer:** This regime prediction analysis is experimental and intended for educational purposes only. "
        "The signals and probabilities shown are based on simple statistical heuristics, not predictive models. "
        "This is not financial advice. Past detection of regime shifts does not guarantee future predictive accuracy."
    )

with tab8:
    st.markdown("### AI Market Analyst")
    st.markdown("*Powered by GPT-4o — contextual analysis of the ChatGPT regime shift*")

    api_key = os.environ.get("OPENAI_API_KEY", "")

    # Section A: Generate AI Analysis
    if st.button("Generate AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing market data with GPT-4o..."):
            context = gather_market_context(
                data, returns, entropy, break_dates, avg_winner_returns, risk_free_rate
            )
            user_message = (
                "Here is the latest data from the ChatGPT Regime Shift Dashboard. "
                "Please provide a concise analysis covering: (1) the current state of the regime shift, "
                "(2) what the early warning signals suggest, and (3) any notable patterns in the data.\n\n"
                + context
            )
            messages = [{"role": "user", "content": user_message}]
            success, response_text = call_openai_analyst(messages, api_key)

            if success:
                st.session_state.ai_chat_history = [
                    {"role": "user", "content": user_message, "hidden": True},
                    {"role": "assistant", "content": response_text},
                ]
            else:
                st.error(response_text)

    # Section B: Chat history display
    for msg in st.session_state.ai_chat_history:
        if msg.get("hidden"):
            continue
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Section C: Follow-up chat
    if len(st.session_state.ai_chat_history) > 0:
        st.markdown("---")
        followup = st.text_input(
            "Ask a follow-up question:",
            key="ai_followup_input",
            placeholder="e.g., How does NVDA's performance compare to previous tech regime shifts?"
        )
        if st.button("Send", key="ai_send_btn") and followup.strip():
            st.session_state.ai_chat_history.append(
                {"role": "user", "content": followup.strip()}
            )
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.ai_chat_history
            ]
            success, response_text = call_openai_analyst(api_messages, api_key)
            if success:
                st.session_state.ai_chat_history.append(
                    {"role": "assistant", "content": response_text}
                )
            else:
                st.session_state.ai_chat_history.append(
                    {"role": "assistant", "content": f"Error: {response_text}"}
                )
            st.rerun()

    # Section D: Disclaimer
    st.markdown("---")
    st.warning(
        "**Disclaimer:** This AI analysis is generated by GPT-4o (OpenAI) for educational purposes as part of "
        "MGMT 69000: Mastering AI for Finance at Purdue University. It is **not financial advice**. The AI may "
        "produce inaccurate or incomplete analysis. Always verify insights against primary data sources."
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>MGMT 69000: Mastering AI for Finance</strong> | Purdue University</p>
    <p>Dashboard built with Streamlit, Plotly, and Python | Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
