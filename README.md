# ChatGPT Regime Shift Analysis Dashboard

**MGMT 69000: Mastering AI for Finance | Purdue University**

An interactive Streamlit dashboard analyzing the ChatGPT Launch (November 30, 2022) as a regime shift event in financial markets, examining creative destruction and sample space expansion. Features 8 analytical tabs including forward-looking regime prediction and GPT-4o-powered market analysis.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📈 Price Performance | Normalized price chart rebased to 100 at ChatGPT launch |
| 📊 Total Returns | Cumulative return bar chart for all tickers |
| 🎯 Market Concentration | Shannon entropy and NVDA market cap share over time |
| 🔄 Regime Detection | CUSUM structural break detection with regime highlighting |
| 🔬 Deep Analysis | Pre/post statistical comparison, correlation heatmaps, Chegg crash deep-dive |
| 💰 Portfolio Impact | $10K portfolio simulator with adjustable start date |
| 🔮 Regime Prediction | Forward-looking CUSUM analysis, historical event validation, early warning signals |
| 🤖 AI Market Analyst | GPT-4o-powered contextual analysis with follow-up chat (via OpenRouter) |

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

# Set up API key for AI Market Analyst tab (optional)
cp .env.example .env
# Edit .env and add your OpenRouter API key

# Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### API Keys

The **🤖 AI Market Analyst** tab requires an [OpenRouter](https://openrouter.ai/) API key to call GPT-4o. All other tabs work without any API key.

1. Sign up at [openrouter.ai](https://openrouter.ai/) and create an API key
2. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=sk-or-v1-your-key-here
   ```
3. The key is loaded automatically via `python-dotenv` at startup

> If no key is configured, the AI tab displays a clear error message — the rest of the dashboard is fully functional.

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
3. **Rolling Shannon Entropy**: Measure market concentration using actual market capitalization (price × shares outstanding)
4. **Regime Detection**: CUSUM analysis to detect structural breaks
5. **Sharpe Ratio Comparison**: Pre/post analysis using live 3-month Treasury yield as risk-free rate
6. **Forward-Looking Prediction**: CUSUM percentile ranking for regime shift probability, validated against historical events (iPhone launch, Bitcoin surge, COVID crash)
7. **Early Warning System**: Entropy trend, correlation structure shift, and volatility regime signals
8. **LLM-Powered Analysis**: GPT-4o contextual market analysis via OpenRouter with conversational follow-up

#### Mathematical Formulations

**Normalized Price:**
```
P_normalized(t) = (P(t) / P(launch_date)) × 100
```

**Shannon Entropy (using actual market cap weights):**
```
w_i = MarketCap_i / Σ MarketCap_j
H = -Σ w_i × log₂(w_i)
```
Where MarketCap_i = Price_i × SharesOutstanding_i. Falls back to price as proxy when shares data is unavailable.

**Concentration Index:**
```
C = 1 / H
```
Higher values indicate more concentration (winner-take-most).

**Sharpe Ratio:**
```
Sharpe = (R_annualized - R_f) / σ_annualized
```
Where R_f is the live 3-month Treasury Bill yield (^IRX), fetched daily from Yahoo Finance.

**Regime Shift Probability (CUSUM Percentile):**
```
S_t = Σ (r_i - μ) / σ           (cumulative sum of standardized returns)
P(shift) = percentile_rank(|S_current|, |S_history|)
```
Ranks the current CUSUM magnitude against the full historical distribution.

---

### I - Implement

#### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive web dashboard |
| Data | yfinance | Yahoo Finance API wrapper |
| Analysis | pandas, numpy | Data manipulation |
| Visualization | Plotly | Interactive charts |
| Statistics | scipy | Regime detection, linear regression |
| LLM Intelligence | OpenAI SDK + OpenRouter | GPT-4o market analysis and chat |
| Configuration | python-dotenv | Environment variable management |

#### Core Functions

```python
# Data fetching with caching (includes shares outstanding for market cap)
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date)

# Risk-free rate from 3-month Treasury
@st.cache_data(ttl=3600)
def fetch_risk_free_rate()

# Entropy using actual market cap weights (price × shares outstanding)
def calculate_rolling_entropy(data, window=60)

# Regime detection via CUSUM
def detect_regime_breaks(returns_series, threshold=2.5)

# Forward-looking regime probability (CUSUM percentile rank)
def calculate_regime_probability(recent_returns, full_returns)

# Early warning signals (entropy trend, correlation shift, volatility regime)
def assess_early_warnings(data, entropy_series)

# Historical event validation for CUSUM detector
def validate_cusum_on_event(returns_series, event_date, threshold=2.0)

# Gather live dashboard data as context for LLM
def gather_market_context(data, returns, entropy, break_dates, ...)

# Call GPT-4o via OpenRouter with error handling
def call_openai_analyst(messages, api_key)
```

#### File Structure

```
chatgpt-regime-shift/
├── app.py              # Main Streamlit application (all tabs, functions, and UI)
├── requirements.txt    # Python dependencies
├── README.md           # This documentation (DRIVER methodology)
├── .gitignore          # Git ignore rules
└── .env                # OpenRouter API key (not committed)
```

#### Key Implementation Decisions

1. **Caching**: Used `@st.cache_data` to avoid redundant API calls (stock data, risk-free rate, historical events)
2. **Interactive Charts**: Plotly over matplotlib for hover, zoom, and unified tooltips
3. **Tab Layout**: 8 tabs progressing from descriptive (tabs 1-4) to analytical (5-6) to predictive (7) to AI-driven (8)
4. **Responsive Design**: Wide layout with sidebar for DRIVER methodology context
5. **Live Risk-Free Rate**: Fetches 3-month Treasury yield (^IRX) for accurate Sharpe ratios instead of hardcoded values
6. **Market Cap Entropy**: Uses actual shares outstanding × price for Shannon entropy, falling back to price when unavailable
7. **Session State for Chat**: `st.session_state.ai_chat_history` persists conversation across Streamlit reruns; resets on new analysis
8. **OpenRouter Routing**: Uses the OpenAI SDK with a custom `base_url` to route through OpenRouter, avoiding direct OpenAI dependency

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

#### Improvements Delivered (Project 2)

- [x] **Fixed entropy calculation** — now uses actual market capitalization (price × shares outstanding) instead of price proxy
- [x] **Fixed Sharpe ratio** — subtracts live 3-month Treasury yield (^IRX) instead of using raw return/volatility
- [x] **Added regime prediction** — forward-looking CUSUM percentile ranking with probability gauge
- [x] **Added historical validation** — CUSUM detector tested against iPhone launch, Bitcoin surge, COVID crash, ChatGPT launch
- [x] **Added early warning system** — entropy trend, correlation structure shift, volatility regime monitoring
- [x] **Added portfolio simulator** — $10K investment comparison across AI Winners, SPY, and CHGG with adjustable dates
- [x] **Added AI Market Analyst** — GPT-4o-powered contextual analysis with conversational follow-up via OpenRouter

#### Remaining Limitations

1. **Limited Universe**: 6 tickers — could expand to include AMD, AVGO, education publishers
2. **No Real-Time Streaming**: Data refreshes on page load (1-hour cache)
3. **Single LLM Provider**: Currently GPT-4o only; could add model selection

#### Future Enhancements

- [ ] Options market implied volatility analysis
- [ ] Sentiment analysis from earnings calls
- [ ] Factor decomposition (separate AI factor from market beta)
- [ ] Cross-asset analysis (bonds, commodities)
- [ ] International markets comparison

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

4. **Sharpe Ratio Accuracy**: Hardcoded risk-free rates introduce silent errors
   - Solution: Live 3-month Treasury yield from ^IRX with fallback default

5. **Market Cap vs Price for Entropy**: Price-based weights distort concentration metrics
   - Solution: Fetch shares outstanding via `yfinance.Ticker.info` and compute actual market cap

6. **Streamlit Session State**: Essential for maintaining LLM conversation history across reruns
   - `st.chat_input` has tab-pinning issues in older Streamlit; `st.text_input` + button is more reliable

7. **OpenRouter as LLM Gateway**: The OpenAI SDK works with any OpenAI-compatible endpoint via `base_url`
   - Avoids vendor lock-in while keeping the same code structure

#### Methodological Insights

1. **Regime Detection Sensitivity**: CUSUM threshold selection is critical
   - Lower threshold = more breaks detected
   - Higher threshold = only major breaks
   - Chose 2.0-2.5 as balance

2. **Entropy Interpretation**: Raw entropy not intuitive
   - Inverse entropy (concentration) more interpretable
   - NVDA share percentage adds concrete context

3. **Historical Validation Matters**: The CUSUM detector successfully flags known events (COVID crash, ChatGPT launch) within days, building confidence in forward-looking predictions

4. **LLM + Structured Data**: Feeding pre-computed dashboard metrics (returns, CUSUM, entropy, warnings) as context produces more grounded analysis than open-ended prompting

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

- OpenAI for ChatGPT (the catalyst studied) and GPT-4o (the analyst powering tab 8)
- [OpenRouter](https://openrouter.ai/) for LLM API routing
- Purdue University MGMT 69000 course
- Yahoo Finance for market data
- Streamlit team for the excellent framework

---

*Built with the DRIVER methodology for quantitative finance research.*
