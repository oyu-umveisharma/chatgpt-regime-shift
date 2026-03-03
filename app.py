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
WINNERS = ['NVDA', 'MSFT', 'META', 'GOOGL', 'AMD', 'AVGO', 'ORCL', 'CRM', 'PLTR']
LOSERS = ['CHGG', 'PRSO', 'TAL', 'UDMY', 'COUR']
BENCHMARK = ['SPY']
ALL_TICKERS = WINNERS + LOSERS + BENCHMARK

# Centralized color scheme
TICKER_COLORS = {
    # Winners (greens, blues, purples)
    'NVDA': '#76B900', 'MSFT': '#00A4EF', 'META': '#0866FF', 'GOOGL': '#4285F4',
    'AMD': '#7B2D8E', 'AVGO': '#1B5E20', 'ORCL': '#00897B', 'CRM': '#1565C0',
    'PLTR': '#5C6BC0',
    # Losers (reds, oranges)
    'CHGG': '#FF6B6B', 'PRSO': '#E53935', 'TAL': '#FF8A65', 'UDMY': '#D32F2F',
    'COUR': '#FF7043',
    # Benchmark
    'SPY': '#888888',
}

# Sector groupings
SECTORS = {
    'AI Infrastructure': ['NVDA', 'AMD', 'AVGO'],
    'Cloud/Enterprise AI': ['MSFT', 'GOOGL', 'ORCL', 'CRM', 'PLTR'],
    'Social/Consumer AI': ['META'],
    'Education (Disrupted)': ['CHGG', 'PRSO', 'TAL', 'UDMY', 'COUR'],
}

SECTOR_COLORS = {
    'AI Infrastructure': '#76B900',
    'Cloud/Enterprise AI': '#00A4EF',
    'Social/Consumer AI': '#0866FF',
    'Education (Disrupted)': '#FF6B6B',
}

def get_ticker_style(ticker):
    """Return (color, dash, width) for a ticker."""
    color = TICKER_COLORS.get(ticker, '#333333')
    if ticker in WINNERS:
        return color, 'solid', 2
    elif ticker in LOSERS:
        return color, 'dash', 2
    else:
        return color, 'dot', 1.5

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
- Tickers tracked — Winners (AI infrastructure): NVDA, MSFT, META, GOOGL, AMD, AVGO, ORCL, CRM, PLTR. \
Losers (disrupted): CHGG, PRSO, TAL, UDMY, COUR. Benchmark: SPY.
- Core thesis: ChatGPT triggered creative destruction (old business models like Chegg, Pearson, and online learning \
platforms disrupted) and sample space expansion (new AI-driven value chains emerged), producing winner-take-most dynamics.

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


def calculate_sector_entropy(data, window=60):
    """Calculate rolling Shannon entropy at the sector level.
    Aggregates market cap by sector, then computes entropy across sectors.
    Returns (sector_entropy_series, sector_caps_df)."""
    # Build per-ticker market cap
    ticker_caps = pd.DataFrame()
    for ticker, df in data.items():
        if ticker == 'SPY':
            continue
        if 'MarketCap' in df.columns:
            ticker_caps[ticker] = df['MarketCap']
        else:
            ticker_caps[ticker] = df['Close']
    ticker_caps = ticker_caps.dropna()

    if ticker_caps.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    # Aggregate to sector level
    sector_caps = pd.DataFrame(index=ticker_caps.index)
    for sector, tickers in SECTORS.items():
        cols = [t for t in tickers if t in ticker_caps.columns]
        if cols:
            sector_caps[sector] = ticker_caps[cols].sum(axis=1)

    def shannon_entropy(weights):
        weights = weights / weights.sum()
        weights = weights[weights > 0]
        return -np.sum(weights * np.log2(weights))

    entropy_series = []
    for i in range(len(sector_caps)):
        if i < window:
            entropy_series.append(np.nan)
        else:
            caps = sector_caps.iloc[i]
            entropy_series.append(shannon_entropy(caps))

    return pd.Series(entropy_series, index=sector_caps.index), sector_caps


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

    for col in normalized_df.columns:
        color, dash, width = get_ticker_style(col)
        fig.add_trace(go.Scatter(
            x=normalized_df.index,
            y=normalized_df[col],
            mode='lines',
            name=col,
            line=dict(color=color, dash=dash, width=width),
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

    # Add traces for each stock
    for col in cumulative.columns:
        color, dash, width = get_ticker_style(col)
        # Emphasize CHGG as the primary crash story
        if col == 'CHGG':
            width = 3
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col],
            mode='lines',
            name=col,
            line=dict(color=color, dash=dash, width=width),
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

    # Strategy 3: Disrupted (equal weight across LOSERS)
    losers_available = [t for t in LOSERS if t in data]
    if losers_available:
        per_stock = initial_investment / len(losers_available)
        portfolio_value = pd.Series(index=all_dates, dtype=float)

        for date in all_dates:
            total = 0
            for ticker in losers_available:
                df = data[ticker]
                if date in df.index:
                    start_idx = df.index.get_indexer([start_date], method='nearest')[0]
                    start_price = df['Close'].iloc[start_idx]
                    current_price = df.loc[date, 'Close']
                    shares = per_stock / start_price
                    total += shares * current_price
            portfolio_value[date] = total

        portfolios['Disrupted Basket'] = portfolio_value

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
        'Disrupted Basket': '#C62828'
    }

    line_styles = {
        'AI Winners': 'solid',
        'SPY Benchmark': 'dot',
        'Disrupted Basket': 'dash'
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
    loser_returns = [returns.get(t, 0) for t in LOSERS if t in returns]
    avg_loser_return = sum(loser_returns) / len(loser_returns) if loser_returns else 0
    st.metric(
        label=f"Disrupted Avg ({len(loser_returns)} stocks)",
        value=f"{avg_loser_return:.1f}%",
        delta="Disrupted Models",
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
    """)

    st.markdown("---")

    # Sector-level entropy analysis
    st.markdown("### Sector-Level Entropy Analysis")
    st.markdown("*Aggregating stocks into sectors for a higher-level view of market concentration*")

    sector_entropy, sector_caps = calculate_sector_entropy(data, window=60)

    if len(sector_entropy.dropna()) > 0 and not sector_caps.empty:
        # Chart 1: Stock-level vs Sector-level entropy comparison
        entropy_compare_fig = go.Figure()

        clean_stock_entropy = entropy.dropna()
        clean_sector_entropy = sector_entropy.dropna()

        if len(clean_stock_entropy) > 0:
            entropy_compare_fig.add_trace(go.Scatter(
                x=clean_stock_entropy.index,
                y=clean_stock_entropy,
                mode='lines',
                name='Stock-Level Entropy',
                line=dict(color='#8B5CF6', width=2),
            ))

        entropy_compare_fig.add_trace(go.Scatter(
            x=clean_sector_entropy.index,
            y=clean_sector_entropy,
            mode='lines',
            name='Sector-Level Entropy',
            line=dict(color='#E65100', width=2, dash='dash'),
        ))

        for date in [CHATGPT_LAUNCH, CHEGG_CRASH, NVIDIA_EARNINGS]:
            entropy_compare_fig.add_vline(x=date, line_dash="dash", line_color="rgba(0,0,0,0.2)")

        entropy_compare_fig.update_layout(
            title=dict(
                text="<b>Stock-Level vs Sector-Level Shannon Entropy</b><br>"
                     "<sup>Lower entropy = higher concentration</sup>",
                font=dict(size=18)
            ),
            xaxis_title="Date",
            yaxis_title="Shannon Entropy (bits)",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(entropy_compare_fig, use_container_width=True)

        # Chart 2: Sector market cap shares — stacked area
        sector_shares = sector_caps.div(sector_caps.sum(axis=1), axis=0) * 100

        stacked_fig = go.Figure()
        for sector in SECTORS:
            if sector in sector_shares.columns:
                stacked_fig.add_trace(go.Scatter(
                    x=sector_shares.index,
                    y=sector_shares[sector],
                    mode='lines',
                    name=sector,
                    line=dict(width=0.5, color=SECTOR_COLORS.get(sector, '#333')),
                    stackgroup='one',
                    hovertemplate=f'{sector}<br>%{{y:.1f}}%<extra></extra>'
                ))

        for date in [CHATGPT_LAUNCH, CHEGG_CRASH, NVIDIA_EARNINGS]:
            stacked_fig.add_vline(x=date, line_dash="dash", line_color="rgba(0,0,0,0.3)")

        stacked_fig.update_layout(
            title=dict(
                text="<b>Sector Market Cap Shares Over Time</b><br>"
                     "<sup>How value distributes across AI sub-sectors</sup>",
                font=dict(size=18)
            ),
            xaxis_title="Date",
            yaxis_title="Market Cap Share (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(stacked_fig, use_container_width=True)

        # Chart 3: Current sector weights — treemap
        latest_caps = sector_caps.iloc[-1]
        latest_total = latest_caps.sum()

        treemap_labels = []
        treemap_parents = []
        treemap_values = []
        treemap_colors = []

        # Root
        treemap_labels.append("All Sectors")
        treemap_parents.append("")
        treemap_values.append(0)
        treemap_colors.append("#FFFFFF")

        for sector in SECTORS:
            if sector in latest_caps.index:
                pct = latest_caps[sector] / latest_total * 100
                treemap_labels.append(f"{sector}<br>{pct:.1f}%")
                treemap_parents.append("All Sectors")
                treemap_values.append(float(latest_caps[sector]))
                treemap_colors.append(SECTOR_COLORS.get(sector, '#333'))

        treemap_fig = go.Figure(go.Treemap(
            labels=treemap_labels,
            parents=treemap_parents,
            values=treemap_values,
            marker=dict(colors=treemap_colors),
            textinfo="label",
            hovertemplate='%{label}<br>Market Cap: $%{value:,.0f}<extra></extra>'
        ))
        treemap_fig.update_layout(
            title=dict(
                text="<b>Current Sector Weights</b>",
                font=dict(size=18)
            ),
            height=400,
            margin=dict(t=50, l=10, r=10, b=10)
        )
        st.plotly_chart(treemap_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Sector Entropy** shows concentration at the **industry level** —
            how value distributes across AI infrastructure, cloud/enterprise,
            social/consumer AI, and disrupted education.
            """)
        with col2:
            st.markdown("""
            **Stock Entropy** shows concentration **within the AI ecosystem** —
            how individual company market caps are distributed. A falling stock
            entropy with stable sector entropy means one company dominates its sector.
            """)
    else:
        st.warning("Insufficient data for sector-level entropy analysis.")

    st.markdown("---")

    st.markdown("""
    *Methodology: Stock-level entropy uses individual market capitalizations
    (price × shares outstanding). Sector-level entropy aggregates stocks into
    four sectors — AI Infrastructure (NVDA, AMD, AVGO), Cloud/Enterprise AI
    (MSFT, GOOGL, ORCL, CRM, PLTR), Social/Consumer AI (META), and Education
    Disrupted (CHGG, PRSO, TAL, UDMY, COUR) — then computes Shannon entropy
    across sector totals. Comparing both levels reveals whether concentration
    is happening between sectors or within them.*
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

    st.markdown("---")

    # Statistical Significance Testing
    st.markdown("### Statistical Significance Testing")
    st.markdown("*Inferential tests confirming whether the ChatGPT regime shift produced statistically significant changes*")

    # Define periods
    pre_start = CHATGPT_LAUNCH - pd.Timedelta(days=180)
    pre_end = CHATGPT_LAUNCH - pd.Timedelta(days=1)
    post_start = CHATGPT_LAUNCH
    post_end = CHATGPT_LAUNCH + pd.Timedelta(days=180)

    # Build per-ticker pre/post daily returns
    test_results = []
    ticker_pre_returns = {}
    ticker_post_returns = {}

    for ticker, df in data.items():
        daily = df['Close'].pct_change().dropna()
        pre = daily[(daily.index >= pre_start) & (daily.index <= pre_end)]
        post = daily[(daily.index >= post_start) & (daily.index <= post_end)]
        if len(pre) >= 20 and len(post) >= 20:
            ticker_pre_returns[ticker] = pre
            ticker_post_returns[ticker] = post

    # --- Test 1: Paired t-test per ticker (pre vs post mean daily return) ---
    st.markdown("#### 1. Pre vs Post Returns by Ticker")
    st.markdown("Paired comparison of daily return distributions before and after ChatGPT launch.")

    per_ticker_rows = []
    for ticker in ALL_TICKERS:
        if ticker not in ticker_pre_returns:
            continue
        pre = ticker_pre_returns[ticker]
        post = ticker_post_returns[ticker]
        # Align to same length for paired test
        min_len = min(len(pre), len(post))
        pre_aligned = pre.values[:min_len]
        post_aligned = post.values[:min_len]

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(post_aligned, pre_aligned)
        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(post_aligned - pre_aligned)
        except ValueError:
            w_stat, w_pval = float('nan'), float('nan')

        category = "Winner" if ticker in WINNERS else ("Loser" if ticker in LOSERS else "Benchmark")
        per_ticker_rows.append({
            'Ticker': ticker,
            'Category': category,
            'Pre Mean (bps)': pre.mean() * 10000,
            'Post Mean (bps)': post.mean() * 10000,
            't-stat': t_stat,
            't p-value': t_pval,
            't Sig': '\u2713' if t_pval < 0.05 else '\u2717',
            'Wilcoxon W': w_stat,
            'W p-value': w_pval,
            'W Sig': '\u2713' if w_pval < 0.05 else '\u2717',
        })

    if per_ticker_rows:
        ticker_test_df = pd.DataFrame(per_ticker_rows)
        st.dataframe(
            ticker_test_df.style.format({
                'Pre Mean (bps)': '{:.1f}',
                'Post Mean (bps)': '{:.1f}',
                't-stat': '{:.3f}',
                't p-value': '{:.4f}',
                'Wilcoxon W': '{:.0f}',
                'W p-value': '{:.4f}',
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Insufficient data for per-ticker tests.")

    # --- Test 2: Winners vs Losers (two-sample t-test) ---
    st.markdown("#### 2. Winners vs Losers: Post-ChatGPT Divergence")
    st.markdown("Two-sample t-test comparing average daily returns of the Winners group vs the Losers group after ChatGPT launch.")

    winner_post_all = []
    loser_post_all = []
    for ticker in WINNERS:
        if ticker in ticker_post_returns:
            winner_post_all.extend(ticker_post_returns[ticker].values)
    for ticker in LOSERS:
        if ticker in ticker_post_returns:
            loser_post_all.extend(ticker_post_returns[ticker].values)

    if len(winner_post_all) >= 20 and len(loser_post_all) >= 20:
        import numpy as np
        winner_arr = np.array(winner_post_all)
        loser_arr = np.array(loser_post_all)

        t2_stat, t2_pval = stats.ttest_ind(winner_arr, loser_arr, equal_var=False)

        group_rows = [{
            'Test': 'Two-Sample t-test (Welch)',
            'Winners Mean (bps)': winner_arr.mean() * 10000,
            'Losers Mean (bps)': loser_arr.mean() * 10000,
            'Difference (bps)': (winner_arr.mean() - loser_arr.mean()) * 10000,
            't-statistic': t2_stat,
            'p-value': t2_pval,
            'Significant': '\u2713' if t2_pval < 0.05 else '\u2717',
        }]

        # Wilcoxon rank-sum (Mann-Whitney U) as non-parametric alternative
        u_stat, u_pval = stats.mannwhitneyu(winner_arr, loser_arr, alternative='two-sided')
        group_rows.append({
            'Test': 'Mann-Whitney U',
            'Winners Mean (bps)': winner_arr.mean() * 10000,
            'Losers Mean (bps)': loser_arr.mean() * 10000,
            'Difference (bps)': (winner_arr.mean() - loser_arr.mean()) * 10000,
            't-statistic': u_stat,
            'p-value': u_pval,
            'Significant': '\u2713' if u_pval < 0.05 else '\u2717',
        })

        group_df = pd.DataFrame(group_rows)
        st.dataframe(
            group_df.style.format({
                'Winners Mean (bps)': '{:.2f}',
                'Losers Mean (bps)': '{:.2f}',
                'Difference (bps)': '{:.2f}',
                't-statistic': '{:.3f}',
                'p-value': '{:.4f}',
            }),
            use_container_width=True,
            hide_index=True
        )

        if t2_pval < 0.05:
            st.success(f"The divergence between Winners and Losers is **statistically significant** "
                       f"(t = {t2_stat:.3f}, p = {t2_pval:.4f}). The regime shift produced meaningfully "
                       f"different return distributions for AI infrastructure vs disrupted companies.")
        else:
            st.info(f"The divergence is **not statistically significant** at the 5% level "
                    f"(t = {t2_stat:.3f}, p = {t2_pval:.4f}).")
    else:
        st.warning("Insufficient data for group comparison test.")

    # Methodology note
    st.markdown("---")
    st.markdown("""
    **Methodology Notes:**
    - **Paired t-test** (`ttest_rel`): Compares pre vs post daily returns for the same stock. Assumes normally distributed differences.
    - **Wilcoxon signed-rank test** (`wilcoxon`): Non-parametric alternative — does not assume normality. Tests whether the median difference is zero.
    - **Two-sample t-test** (`ttest_ind`, Welch's): Compares Winners vs Losers group returns post-launch. Does not assume equal variance.
    - **Mann-Whitney U** (`mannwhitneyu`): Non-parametric alternative to the two-sample t-test.
    - **p < 0.05** indicates a statistically significant difference at the 5% level.
    - Returns are expressed in **basis points (bps)** where 1 bps = 0.01%.
    """)

    st.markdown("---")

    # ===== Sample Space Expansion Analysis =====
    st.markdown("### Sample Space Expansion")
    st.markdown("*Directly measuring whether ChatGPT created a new investable asset class*")

    # Build combined AI winners market cap over time
    ai_market_cap = pd.DataFrame()
    for ticker in WINNERS:
        if ticker in data:
            df = data[ticker]
            if 'MarketCap' in df.columns:
                ai_market_cap[ticker] = df['MarketCap']
            else:
                ai_market_cap[ticker] = df['Close']
    ai_market_cap = ai_market_cap.dropna()

    spy_market_cap = None
    if 'SPY' in data:
        spy_df = data['SPY']
        if 'MarketCap' in spy_df.columns:
            spy_market_cap = spy_df['MarketCap']
        else:
            spy_market_cap = spy_df['Close']

    if not ai_market_cap.empty:
        total_ai_cap = ai_market_cap.sum(axis=1)

        # --- Pre vs Post metrics ---
        pre_idx = ai_market_cap.index.get_indexer([CHATGPT_LAUNCH], method='nearest')[0]
        pre_cap = total_ai_cap.iloc[pre_idx]
        post_cap = total_ai_cap.iloc[-1]
        growth_multiple = post_cap / pre_cap if pre_cap > 0 else 0

        # Count stocks above $100B, $500B, $1T at launch vs now
        pre_caps = ai_market_cap.iloc[pre_idx]
        post_caps = ai_market_cap.iloc[-1]

        # Use actual market cap if available, else skip threshold counts
        has_real_mcap = any('MarketCap' in data[t].columns for t in WINNERS if t in data)

        if has_real_mcap:
            pre_100b = int((pre_caps > 100e9).sum())
            post_100b = int((post_caps > 100e9).sum())
            pre_500b = int((pre_caps > 500e9).sum())
            post_500b = int((post_caps > 500e9).sum())
            pre_1t = int((pre_caps > 1e12).sum())
            post_1t = int((post_caps > 1e12).sum())

        # AI as % of SPY (proxy for market weight)
        pre_ai_pct = None
        post_ai_pct = None
        if spy_market_cap is not None and len(spy_market_cap) > pre_idx:
            # SPY price * ~500 constituents is not true market cap,
            # so we use ratio change as a relative measure
            spy_pre = spy_market_cap.iloc[pre_idx]
            spy_post = spy_market_cap.iloc[-1]
            if spy_pre > 0 and spy_post > 0:
                pre_ai_pct = (pre_cap / spy_pre)
                post_ai_pct = (post_cap / spy_post)

        # Sample Space Expansion Score
        expansion_score = growth_multiple

        # --- Display metrics row ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="AI Sector Growth Multiple",
                value=f"{growth_multiple:.1f}x",
                delta=f"${pre_cap/1e12:.1f}T → ${post_cap/1e12:.1f}T"
            )
        with m2:
            if has_real_mcap:
                st.metric(
                    label="AI Stocks > $100B Market Cap",
                    value=f"{post_100b}",
                    delta=f"+{post_100b - pre_100b} since launch"
                )
            else:
                st.metric(
                    label="AI Sector Total",
                    value=f"${post_cap/1e12:.1f}T",
                    delta=f"+{(growth_multiple - 1) * 100:.0f}%"
                )
        with m3:
            if pre_ai_pct is not None:
                change = ((post_ai_pct / pre_ai_pct) - 1) * 100
                st.metric(
                    label="AI/SPY Ratio Change",
                    value=f"+{change:.0f}%",
                    delta="AI grew faster than the market"
                )
            else:
                st.metric(
                    label="Expansion Score",
                    value=f"{expansion_score:.1f}x",
                    delta="Post/pre market cap ratio"
                )

        # --- Pre vs Post comparison table ---
        st.markdown("#### Pre vs Post ChatGPT Comparison")

        comparison_rows = [
            {
                'Metric': 'AI Sector Total Market Cap',
                'Pre-ChatGPT': f'${pre_cap/1e12:.2f}T',
                'Post-ChatGPT (Current)': f'${post_cap/1e12:.2f}T',
                'Change': f'+{(growth_multiple - 1) * 100:.0f}%',
            },
        ]

        if has_real_mcap:
            comparison_rows.extend([
                {
                    'Metric': 'AI Stocks > $100B Market Cap',
                    'Pre-ChatGPT': str(pre_100b),
                    'Post-ChatGPT (Current)': str(post_100b),
                    'Change': f'+{post_100b - pre_100b}',
                },
                {
                    'Metric': 'AI Stocks > $500B Market Cap',
                    'Pre-ChatGPT': str(pre_500b),
                    'Post-ChatGPT (Current)': str(post_500b),
                    'Change': f'+{post_500b - pre_500b}',
                },
                {
                    'Metric': 'AI Stocks > $1T Market Cap',
                    'Pre-ChatGPT': str(pre_1t),
                    'Post-ChatGPT (Current)': str(post_1t),
                    'Change': f'+{post_1t - pre_1t}',
                },
            ])

        if pre_ai_pct is not None:
            comparison_rows.append({
                'Metric': 'AI/SPY Ratio (relative weight)',
                'Pre-ChatGPT': f'{pre_ai_pct:.1f}',
                'Post-ChatGPT (Current)': f'{post_ai_pct:.1f}',
                'Change': f'+{((post_ai_pct / pre_ai_pct) - 1) * 100:.0f}%',
            })

        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # --- Chart: Combined AI market cap over time with threshold annotations ---
        st.markdown("#### AI Sector Market Cap Expansion")

        expansion_fig = go.Figure()

        # Stacked area by ticker
        for ticker in WINNERS:
            if ticker in ai_market_cap.columns:
                color, _, _ = get_ticker_style(ticker)
                expansion_fig.add_trace(go.Scatter(
                    x=ai_market_cap.index,
                    y=ai_market_cap[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(width=0.5, color=color),
                    stackgroup='one',
                    hovertemplate=f'{ticker}<br>${{y:,.0f}}<extra></extra>'
                ))

        # Add threshold lines
        thresholds = [(1e12, '$1T'), (2e12, '$2T'), (5e12, '$5T'), (10e12, '$10T')]
        for threshold, label in thresholds:
            if total_ai_cap.max() >= threshold:
                expansion_fig.add_hline(
                    y=threshold, line_dash="dot", line_color="rgba(0,0,0,0.3)",
                    annotation_text=label, annotation_position="right"
                )

                # Find first date crossing
                crossing = total_ai_cap[total_ai_cap >= threshold]
                if not crossing.empty:
                    cross_date = crossing.index[0]
                    expansion_fig.add_annotation(
                        x=cross_date, y=threshold,
                        text=f"Crossed {label}<br>{cross_date.strftime('%b %Y')}",
                        showarrow=True, arrowhead=2, arrowcolor="#666",
                        font=dict(size=9), bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#666", borderwidth=1, borderpad=3
                    )

        # ChatGPT launch line
        expansion_fig.add_vline(x=CHATGPT_LAUNCH, line_dash="dash", line_color="blue", line_width=1.5)
        expansion_fig.add_annotation(
            x=CHATGPT_LAUNCH, y=total_ai_cap.max() * 0.95,
            text="ChatGPT Launch", showarrow=False,
            font=dict(size=10, color="blue"), bgcolor="rgba(255,255,255,0.8)"
        )

        expansion_fig.update_layout(
            title=dict(
                text="<b>AI Sector Market Cap: Sample Space Expansion</b><br>"
                     "<sup>Combined market capitalization of AI winner stocks over time</sup>",
                font=dict(size=18)
            ),
            xaxis_title="Date",
            yaxis_title="Market Cap ($)",
            yaxis=dict(tickformat='$,.0s'),
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(expansion_fig, use_container_width=True)

        # --- Historical expansion comparison ---
        st.markdown("#### Historical Context: Technology Regime Shifts")

        st.markdown("""
        | Regime Shift | Period | Sample Space Expansion |
        |-------------|--------|----------------------|
        | **Dot-Com / Internet** | 1995-2000 | Internet stocks grew from <1% to ~6% of S&P 500; dozens of new IPOs created a new "internet sector" |
        | **Mobile / Smartphone** | 2007-2012 | Apple grew from $100B to $500B; mobile app economy created entirely new revenue streams |
        | **ChatGPT / AI** | 2022-Present | AI infrastructure sector grew **{growth_multiple:.1f}x**; NVIDIA alone added ~$2T+ in market cap |

        The ChatGPT regime shift shows the hallmarks of a **sample space expansion event**:
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What expanded:**
            - AI infrastructure became a distinct investable theme
            - GPU/accelerator demand created new supply chain valuations
            - Enterprise AI (CRM, PLTR, ORCL) revalued as AI plays
            - AI-related ETFs proliferated (BOTZ, AIQ, ROBT)
            """)
        with col2:
            st.markdown("""
            **What contracted:**
            - Education tech models commoditized by free AI
            - Homework help services lost competitive moats
            - Content creation platforms face AI substitution
            - Traditional tutoring disrupted globally
            """)
    else:
        st.warning("Insufficient market cap data for sample space analysis.")

with tab6:
    st.markdown("### Portfolio Impact Simulator")
    st.markdown("*Compare investment strategies: AI Winners vs Market Benchmark vs Disrupted Stock*")

    # Simplified date selection
    DATE_OPTIONS = {
        "ChatGPT Launch (Nov 30, 2022)": pd.Timestamp('2022-11-30'),
        "Start of 2023 (Jan 3)": pd.Timestamp('2023-01-03'),
        "Post-Chegg Crash (Jun 2023)": pd.Timestamp('2023-06-01'),
        "Start of 2024 (Jan 2)": pd.Timestamp('2024-01-02'),
        "Start of 2025 (Jan 2)": pd.Timestamp('2025-01-02'),
    }

    selected_label = st.select_slider(
        "Select investment start date:",
        options=list(DATE_OPTIONS.keys()),
        value="ChatGPT Launch (Nov 30, 2022)",
    )

    selected_timestamp = DATE_OPTIONS[selected_label]

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
            disrupted_return = summary_df[summary_df['Strategy'] == 'Disrupted Basket']['Return_numeric'].values

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
                if len(ai_return) > 0 and len(disrupted_return) > 0:
                    spread = ai_return[0] - disrupted_return[0]
                    st.metric(
                        "Winner vs Loser Spread",
                        f"{spread:+.1f}%",
                        delta="Creative Destruction Gap"
                    )

            with col3:
                if len(disrupted_return) > 0:
                    st.metric(
                        "Disruption Impact",
                        f"{disrupted_return[0]:+.1f}%",
                        delta="Value Destroyed",
                        delta_color="inverse"
                    )

        st.markdown("""
        **Investment Lessons from the AI Regime Shift:**
        - **Timing matters, but direction matters more**: Even investing after the initial surge, AI infrastructure outperformed
        - **Disruption is permanent**: Disrupted companies (CHGG, education platforms) never recovered - these weren't dips to buy
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
