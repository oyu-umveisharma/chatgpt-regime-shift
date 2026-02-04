# ChatGPT Regime Shift Analysis Dashboard

**MGMT 69000: Mastering AI for Finance | Purdue University**

An interactive Streamlit dashboard analyzing the ChatGPT Launch (November 30, 2022) as a regime shift event in financial markets, examining creative destruction and sample space expansion.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/chatgpt-regime-shift.git
cd chatgpt-regime-shift

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## DRIVER Methodology Documentation

This project follows the **DRIVER** framework for quantitative finance research and product development.

### D - Discover

#### Research Question
How did the ChatGPT launch restructure market valuations across the AI value chain, and can we identify this as a regime shift event?

#### Background Research
- **November 30, 2022**: OpenAI releases ChatGPT
- Within 5 days, ChatGPT reaches 1 million users
- By January 2023, it becomes the fastest-growing consumer application in history
- Market immediately begins repricing AI-related assets

#### Hypotheses
1. **Creative Destruction Hypothesis**: ChatGPT would cause fundamental damage to business models relying on information arbitrage (education tech, content creation)
2. **Sample Space Expansion Hypothesis**: A new asset class ("AI infrastructure") would emerge and create unprecedented returns
3. **Winner-Take-Most Hypothesis**: Returns would concentrate in a small number of infrastructure providers

#### Data Sources Identified
- Yahoo Finance API (yfinance) for historical price data
- Market capitalization data for weight calculations
- Event dates from news sources

---

### R - Represent

#### Stock Universe Selection

| Category | Ticker | Rationale |
|----------|--------|-----------|
| **Winner** | NVDA | GPU monopoly, AI compute infrastructure |
| **Winner** | MSFT | OpenAI partnership, Azure AI services |
| **Winner** | META | AI research, LLaMA models, ad optimization |
| **Winner** | GOOGL | AI research heritage, Gemini, cloud AI |
| **Loser** | CHGG | Education tech, homework help disrupted |
| **Benchmark** | SPY | S&P 500 market benchmark |

#### Key Dates

| Date | Event | Significance |
|------|-------|--------------|
| Nov 30, 2022 | ChatGPT Launch | Regime shift trigger |
| May 2, 2023 | Chegg Crash | -49% single day, creative destruction |
| May 24, 2023 | NVIDIA Earnings | AI demand confirmation |

#### Analytical Framework

1. **Normalized Price Chart**: Rebase all prices to 100 at ChatGPT launch to visualize relative performance
2. **Total Returns**: Bar chart showing cumulative returns since launch
3. **Rolling Shannon Entropy**: Measure market concentration changes
4. **Regime Detection**: CUSUM analysis to detect structural breaks

#### Mathematical Formulations

**Normalized Price:**
```
P_normalized(t) = (P(t) / P(launch_date)) × 100
```

**Shannon Entropy:**
```
H = -Σ p_i × log₂(p_i)
```
Where p_i is the weight of stock i in the portfolio.

**Concentration Index:**
```
C = 1 / H
```
Higher values indicate more concentration (winner-take-most).

---

### I - Implement

#### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive web dashboard |
| Data | yfinance | Yahoo Finance API wrapper |
| Analysis | pandas, numpy | Data manipulation |
| Visualization | Plotly | Interactive charts |
| Statistics | scipy | Regime detection |

#### Core Functions

```python
# Data fetching with caching
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""

# Price normalization
def normalize_prices(data, base_date):
    """Normalize all prices to 100 at the base date."""

# Entropy calculation
def calculate_rolling_entropy(data, window=60):
    """Calculate rolling Shannon entropy of market cap weights."""

# Regime detection
def detect_regime_breaks(returns_series, threshold=2.5):
    """Detect structural break points using CUSUM analysis."""
```

#### File Structure

```
chatgpt-regime-shift/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This documentation
├── .gitignore         # Git ignore rules
└── .env               # Environment variables (not committed)
```

#### Key Implementation Decisions

1. **Caching**: Used `@st.cache_data` to avoid redundant API calls
2. **Interactive Charts**: Plotly over matplotlib for better user experience
3. **Tab Layout**: Organized analysis into logical sections
4. **Responsive Design**: Wide layout with sidebar for context

---

### V - Validate

#### Quantitative Validation

| Metric | Expected | Observed | Status |
|--------|----------|----------|--------|
| NVDA outperforms SPY | Yes | +800%+ vs ~40% | ✅ Confirmed |
| CHGG negative returns | Yes | -90%+ decline | ✅ Confirmed |
| Entropy decreases | Yes | Winner concentration | ✅ Confirmed |
| Regime breaks detected | Yes | Multiple breaks found | ✅ Confirmed |

#### Hypothesis Testing Results

**H1: Creative Destruction** ✅ CONFIRMED
- Chegg lost 49% in a single day (May 2, 2023)
- Total decline exceeds 90% from pre-ChatGPT levels
- CEO explicitly cited ChatGPT in earnings call

**H2: Sample Space Expansion** ✅ CONFIRMED
- "AI infrastructure" emerged as distinct investment theme
- NVDA market cap grew from ~$400B to ~$3T+
- New ETFs created specifically for AI theme

**H3: Winner-Take-Most** ✅ CONFIRMED
- NVDA dominates AI winners basket
- Shannon entropy decreased over time
- Concentration index increased post-launch

#### Visual Validation

The dashboard provides visual confirmation:
1. **Divergence Chart**: Clear separation between winners/losers
2. **Returns Bar**: Magnitude of outperformance visible
3. **Entropy Plot**: Declining entropy trend clear
4. **Regime Chart**: Structural breaks identified at key dates

---

### E - Evolve

#### Current Limitations

1. **Price-Based Weights**: Using price as proxy for market cap weights
2. **Limited Universe**: Only 6 tickers analyzed
3. **Static Analysis**: No real-time streaming

#### Proposed Enhancements

##### Short-term (Next Sprint)
- [ ] Add more disrupted companies (educational publishers, content farms)
- [ ] Include actual market cap data for proper entropy calculation
- [ ] Add volatility regime analysis

##### Medium-term (Next Quarter)
- [ ] Options market implied volatility analysis
- [ ] Sentiment analysis from earnings calls
- [ ] Factor decomposition (separate AI factor from market beta)

##### Long-term (Future Research)
- [ ] Cross-asset analysis (bonds, commodities)
- [ ] International markets comparison
- [ ] Real-time regime monitoring system

#### Extensibility Design

The codebase is designed for easy extension:
```python
# Add new tickers
WINNERS = ['NVDA', 'MSFT', 'META', 'GOOGL', 'AMD', 'AVGO']
LOSERS = ['CHGG', 'PRSO', 'UPWK']  # Add more disrupted companies

# Add new events
EVENTS = [
    (pd.Timestamp('2022-11-30'), "ChatGPT Launch"),
    (pd.Timestamp('2023-03-14'), "GPT-4 Release"),
    (pd.Timestamp('2023-05-02'), "Chegg Crash"),
    # Add more events...
]
```

---

### R - Reflect

#### Technical Learnings

1. **yfinance Reliability**: Generally reliable but occasional timeout issues
   - Solution: Implemented caching and error handling

2. **Streamlit Performance**: Heavy computations can slow UI
   - Solution: Aggressive use of `@st.cache_data`

3. **Plotly Annotations**: Complex annotation positioning
   - Solution: Dynamic y-position calculation

#### Methodological Insights

1. **Regime Detection Sensitivity**: CUSUM threshold selection is critical
   - Lower threshold = more breaks detected
   - Higher threshold = only major breaks
   - Chose 2.0-2.5 as balance

2. **Entropy Interpretation**: Raw entropy not intuitive
   - Inverse entropy (concentration) more interpretable
   - NVDA share percentage adds concrete context

#### Market Understanding Gained

1. **Speed of Repricing**: Markets repriced entire value chains within months
2. **Second-Order Effects**: Not just OpenAI benefited - entire infrastructure ecosystem
3. **Non-Linear Impacts**: Losers didn't just underperform; some faced existential threats

#### Framework Effectiveness

The DRIVER framework proved valuable for:
- Structured research process
- Clear documentation
- Reproducible analysis
- Easy communication to stakeholders

---

## Academic Context

### Course Information
- **Course**: MGMT 69000 - Mastering AI for Finance
- **Institution**: Purdue University, Daniels School of Business
- **Topic**: Regime Shifts and Sample Space Expansion in Financial Markets

### Key Concepts Demonstrated

1. **Regime Shifts**: Fundamental changes in market dynamics
2. **Creative Destruction**: Schumpeterian disruption of existing business models
3. **Sample Space Expansion**: Emergence of new asset classes and return distributions
4. **Winner-Take-Most**: Concentration dynamics in technology markets

### Further Reading

- Sornette, D. (2003). "Why Stock Markets Crash: Critical Events in Complex Financial Systems"
- Taleb, N.N. (2007). "The Black Swan: The Impact of the Highly Improbable"
- Arthur, W.B. (2009). "The Nature of Technology: What It Is and How It Evolves"

---

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- OpenAI for ChatGPT (the catalyst studied)
- Purdue University MGMT 69000 course
- Yahoo Finance for market data
- Streamlit team for the excellent framework

---

*Built with the DRIVER methodology for quantitative finance research.*
