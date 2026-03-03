# ChatGPT Regime Shift Analysis Dashboard — Final Project

![CI](https://github.com/oyu-umveisharma/chatgpt-regime-shift/actions/workflows/ci.yml/badge.svg)
![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![Tests](https://img.shields.io/badge/Tests-21%20Passing-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-green)

**MGMT 69000: Mastering AI for Finance | Purdue University | Daniels School of Business**

A CI/CD-validated, production-grade Streamlit dashboard analyzing the ChatGPT Launch (November 30, 2022) as a regime shift event in financial markets. Tracks 15 stocks across 4 sectors, applies inferential statistical testing, measures sample space expansion directly, and integrates GPT-4o-powered market analysis — all backed by 21 automated unit tests running on every commit.

> **[AI Collaboration Log](AI_LOG.md)** — Full documentation of the AI-assisted development process, prompts used, code generated, and validation steps for each Final Project feature.

---

## Features at a Glance

### 15-Ticker Universe Across 4 Sectors

| Sector | Tickers | Thesis |
|--------|---------|--------|
| **AI Infrastructure** | NVDA, AMD, AVGO | GPU compute, networking, custom accelerators |
| **Cloud/Enterprise AI** | MSFT, GOOGL, ORCL, CRM, PLTR | Cloud platforms, enterprise AI adoption |
| **Social/Consumer AI** | META | LLaMA models, AI-driven ad optimization |
| **Education (Disrupted)** | CHGG, PRSO, TAL, UDMY, COUR | Business models disrupted by ChatGPT |
| **Benchmark** | SPY | S&P 500 market reference |

### 8 Analytical Tabs

| Tab | Description |
|-----|-------------|
| **Price Performance** | Normalized price chart rebased to 100 at ChatGPT launch for all 15 tickers |
| **Total Returns** | Cumulative return bar chart with winner/loser/benchmark color coding |
| **Market Concentration** | Stock-level and sector-level Shannon entropy, stacked area charts, treemap |
| **Regime Detection** | CUSUM structural break detection with regime highlighting |
| **Deep Analysis** | Pre/post statistical comparison, inferential tests, correlation heatmaps, sample space expansion, Chegg crash deep-dive |
| **Portfolio Impact** | $10K portfolio simulator with 5 preset start dates |
| **Regime Prediction** | Forward-looking CUSUM probability, historical event validation, early warning signals |
| **AI Market Analyst** | GPT-4o-powered contextual analysis with conversational follow-up (via OpenRouter) |

### Inferential Statistical Testing

Four rigorous tests confirm the regime shift is statistically significant:

| Test | Method | Purpose |
|------|--------|---------|
| Paired t-test | `scipy.stats.ttest_rel` | Pre vs post daily returns per ticker (parametric) |
| Wilcoxon signed-rank | `scipy.stats.wilcoxon` | Pre vs post daily returns per ticker (non-parametric) |
| Welch's t-test | `scipy.stats.ttest_ind` | Winners vs Losers group comparison (unequal variance) |
| Mann-Whitney U | `scipy.stats.mannwhitneyu` | Winners vs Losers group comparison (non-parametric) |

Results: Group tests confirm statistically significant Winner/Loser divergence post-ChatGPT (p < 0.05).

### Sector-Level Shannon Entropy

Goes beyond individual stock analysis to measure concentration at the industry level:

- **Stock-level vs sector-level entropy comparison** — reveals whether concentration is happening *between* sectors or *within* them
- **Stacked area chart** — sector market cap shares over time
- **Treemap** — current sector weight distribution

### Sample Space Expansion Measurement

Directly tests the hypothesis that ChatGPT created a new market regime:

- AI sector market cap growth multiples (pre-launch vs current)
- Threshold crossing timeline ($100B, $500B, $1T, $5T, $10T)
- Historical comparison to dot-com bubble and mobile revolution regime shifts

### Automated Validation & CI/CD

Every push triggers a GitHub Actions pipeline:

| Step | Tool | Description |
|------|------|-------------|
| Syntax Check | `py_compile` | Validates `app.py` compiles without errors |
| Unit Tests | `pytest` | Runs 21 tests across 7 test classes |
| Linting | `flake8` | Code style checks (warnings only, non-blocking) |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/oyu-umveisharma/chatgpt-regime-shift.git
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

The **AI Market Analyst** tab requires an [OpenRouter](https://openrouter.ai/) API key to call GPT-4o. All other 7 tabs work without any API key.

1. Sign up at [openrouter.ai](https://openrouter.ai/) and create an API key
2. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=sk-or-v1-your-key-here
   ```
3. The key is loaded automatically via `python-dotenv` at startup

> If no key is configured, the AI tab displays a clear error message — the rest of the dashboard is fully functional.

---

## DRIVER Methodology

This project follows the **DRIVER** framework (Discover, Represent, Implement, Validate, Evolve, Reflect) for quantitative finance research and product development. Each phase below documents how the methodology guided development from initial prototype through final project delivery.

### D — Discover

#### Research Question
How did the ChatGPT launch restructure market valuations across the AI value chain, and can we identify this as a statistically significant regime shift event?

#### Background Research
- **November 30, 2022**: OpenAI releases ChatGPT
- Within 5 days, ChatGPT reaches 1 million users
- By January 2023, it becomes the fastest-growing consumer application in history
- Market immediately begins repricing AI-related assets across the entire value chain

#### Hypotheses
1. **Creative Destruction Hypothesis**: ChatGPT would cause fundamental damage to business models relying on information arbitrage (education tech, content creation)
2. **Sample Space Expansion Hypothesis**: A new asset class ("AI infrastructure") would emerge and create unprecedented returns
3. **Winner-Take-Most Hypothesis**: Returns would concentrate in a small number of infrastructure providers

#### Data Sources
- Yahoo Finance API (yfinance) for historical price data and shares outstanding
- 3-month Treasury Bill yield (^IRX) for risk-free rate
- Market capitalization data for entropy and concentration calculations
- Event dates from news sources and earnings calls

---

### R — Represent

#### Stock Universe (15 Tickers, 4 Sectors)

| Category | Ticker | Rationale |
|----------|--------|-----------|
| **AI Infrastructure** | NVDA | GPU monopoly, AI compute infrastructure |
| **AI Infrastructure** | AMD | AI chip competitor, data center GPUs |
| **AI Infrastructure** | AVGO | Broadcom — AI networking, custom accelerators |
| **Cloud/Enterprise AI** | MSFT | OpenAI partnership, Azure AI services |
| **Cloud/Enterprise AI** | GOOGL | AI research heritage, Gemini, cloud AI |
| **Cloud/Enterprise AI** | ORCL | Oracle — cloud AI infrastructure |
| **Cloud/Enterprise AI** | CRM | Salesforce — enterprise AI (Einstein, Agentforce) |
| **Cloud/Enterprise AI** | PLTR | Palantir — AI analytics, government/enterprise AI |
| **Social/Consumer AI** | META | AI research, LLaMA models, ad optimization |
| **Education (Disrupted)** | CHGG | Chegg — homework help disrupted by ChatGPT |
| **Education (Disrupted)** | PRSO | Pearson — traditional education publisher |
| **Education (Disrupted)** | TAL | TAL Education — tutoring services disrupted |
| **Education (Disrupted)** | UDMY | Udemy — online learning marketplace |
| **Education (Disrupted)** | COUR | Coursera — online course platform |
| **Benchmark** | SPY | S&P 500 market benchmark |

*Expanded from 6 tickers in Project 2 to 15 tickers for the Final Project based on professor feedback to broaden the analysis across AI sub-sectors and disrupted industries.*

#### Key Dates

| Date | Event | Significance |
|------|-------|--------------|
| Nov 30, 2022 | ChatGPT Launch | Regime shift trigger |
| May 2, 2023 | Chegg Crash | -49% single day, creative destruction confirmed |
| May 24, 2023 | NVIDIA Earnings | AI infrastructure demand validated |

#### Analytical Framework

| # | Analysis | Method |
|---|----------|--------|
| 1 | Normalized Price Chart | Rebase all prices to 100 at ChatGPT launch |
| 2 | Total Returns | Cumulative returns bar chart since launch |
| 3 | Market Concentration | Stock-level and sector-level Shannon entropy using actual market capitalization |
| 4 | Regime Detection | CUSUM analysis to detect structural breaks |
| 5 | Sharpe Ratio Comparison | Pre/post analysis using live 3-month Treasury yield |
| 6 | Forward-Looking Prediction | CUSUM percentile ranking validated against iPhone, Bitcoin, COVID, ChatGPT |
| 7 | Early Warning System | Entropy trend, correlation structure shift, volatility regime signals |
| 8 | Inferential Testing | Paired t-test, Wilcoxon, Welch's t-test, Mann-Whitney U |
| 9 | Sample Space Expansion | Market cap growth multiples, threshold crossings, historical comparison |
| 10 | LLM Analysis | GPT-4o contextual market analysis via OpenRouter |

#### Mathematical Formulations

**Normalized Price:**
```
P_normalized(t) = (P(t) / P(launch_date)) × 100
```

**Shannon Entropy (stock-level, using actual market cap):**
```
w_i = MarketCap_i / Σ MarketCap_j
H = -Σ w_i × log₂(w_i)
```
Where `MarketCap_i = Price_i × SharesOutstanding_i`. Falls back to price as proxy when shares data is unavailable.

**Shannon Entropy (sector-level):**
```
SectorCap_s = Σ MarketCap_i  (for all tickers i in sector s)
w_s = SectorCap_s / Σ SectorCap_j
H_sector = -Σ w_s × log₂(w_s)
```

Comparing stock-level vs sector-level entropy reveals whether concentration is happening *between* sectors or *within* them.

**Concentration Index:**
```
C = 1 / H
```
Higher values indicate more concentration (winner-take-most dynamics).

**Sharpe Ratio:**
```
Sharpe = (R_annualized - R_f) / σ_annualized
```
Where `R_f` is the live 3-month Treasury Bill yield (^IRX), fetched daily from Yahoo Finance.

**Regime Shift Probability (CUSUM Percentile):**
```
S_t = Σ (r_i - μ) / σ           (cumulative sum of standardized returns)
P(shift) = percentile_rank(|S_current|, |S_history|)
```
Ranks the current CUSUM magnitude against the full historical distribution.

**Inferential Tests:**

| Test | Function | Purpose |
|------|----------|---------|
| Paired t-test | `scipy.stats.ttest_rel` | Pre vs post daily returns per ticker (assumes normality) |
| Wilcoxon signed-rank | `scipy.stats.wilcoxon` | Non-parametric alternative to paired t-test |
| Welch's t-test | `scipy.stats.ttest_ind` | Winners vs Losers group comparison (unequal variance) |
| Mann-Whitney U | `scipy.stats.mannwhitneyu` | Non-parametric alternative to two-sample t-test |

- p < 0.05 indicates statistical significance at the 5% level
- Returns expressed in basis points (bps); 1 bps = 0.01%
- Group tests confirm statistically significant Winner/Loser divergence post-ChatGPT

---

### I — Implement

#### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive web dashboard with 8 tabs |
| Data | yfinance | Yahoo Finance API for prices, shares outstanding, Treasury yield |
| Analysis | pandas, numpy | Data manipulation and numerical computation |
| Visualization | Plotly | Interactive charts with hover, zoom, unified tooltips |
| Statistics | scipy | Regime detection (CUSUM), inferential tests (t-test, Wilcoxon, Mann-Whitney) |
| LLM Intelligence | OpenAI SDK + OpenRouter | GPT-4o market analysis and conversational follow-up |
| Configuration | python-dotenv | Environment variable management |
| Testing | pytest | 21 unit tests across 7 test classes |
| Linting | flake8 | Code style validation |
| CI/CD | GitHub Actions | Automated syntax check, testing, and linting on every push/PR |

#### Core Functions

```python
# Data fetching with caching (includes shares outstanding for market cap)
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date)

# Risk-free rate from 3-month Treasury
@st.cache_data(ttl=3600)
def fetch_risk_free_rate()

# Stock-level entropy using actual market cap weights
def calculate_rolling_entropy(data, window=60)

# Sector-level entropy across 4 industry groups
def calculate_sector_entropy(data, window=60)

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
├── app.py                          # Main Streamlit application (~2200 lines)
├── requirements.txt                # Python dependencies (10 packages)
├── README.md                       # DRIVER methodology documentation
├── AI_LOG.md                       # AI collaboration log (required deliverable)
├── validate.py                     # Manual validation script
├── .gitignore                      # Git ignore rules
├── .env                            # OpenRouter API key (not committed)
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
└── tests/
    ├── __init__.py
    └── test_app.py                 # 21 unit tests across 7 test classes
```

#### Key Implementation Decisions

1. **Caching**: `@st.cache_data(ttl=3600)` avoids redundant API calls for stock data, risk-free rate, and historical events
2. **Interactive Charts**: Plotly over matplotlib for hover, zoom, and unified tooltips across all 15 tickers
3. **Tab Layout**: 8 tabs progressing from descriptive (1-4) to analytical (5-6) to predictive (7) to AI-driven (8)
4. **Live Risk-Free Rate**: Fetches 3-month Treasury yield (^IRX) for accurate Sharpe ratios instead of hardcoded values
5. **Market Cap Entropy**: Uses actual shares outstanding × price for Shannon entropy, falling back to price when unavailable
6. **Centralized Color Scheme**: `TICKER_COLORS` dict and `get_ticker_style()` function ensure consistent visual identity across all charts
7. **Session State for Chat**: `st.session_state.ai_chat_history` persists conversation across Streamlit reruns; resets on new analysis
8. **OpenRouter Routing**: Uses the OpenAI SDK with a custom `base_url` to route through OpenRouter, avoiding direct OpenAI dependency
9. **Simplified Date Selection**: Portfolio simulator uses `st.select_slider` with 5 meaningful preset dates instead of a complex continuous slider

---

## Validation & CI/CD

### GitHub Actions Pipeline

Every push and pull request to `main` triggers an automated CI pipeline:

```yaml
Workflow: CI
Trigger:  push/PR to main
Runner:   ubuntu-latest, Python 3.11
```

| Step | Tool | Description | Blocking |
|------|------|-------------|----------|
| Syntax Check | `py_compile` | Validates `app.py` compiles without errors | Yes |
| Unit Tests | `pytest` | Runs 21 tests across 7 test classes | Yes |
| Linting | `flake8` | Code style checks (max line length 120) | No (warnings only) |

### Test Suite — 21 Tests, 7 Classes

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestFetchRiskFreeRate` | 2 | Returns float, value between 0-20% |
| `TestCalculateReturns` | 3 | Dict output, numeric values, positive drift produces positive return |
| `TestCalculateRollingEntropy` | 2 | Returns Series, values within [0, log₂(n)] bounds |
| `TestDetectRegimeBreaks` | 4 | Returns list, detects known breaks, Timestamp types, handles short series |
| `TestStatisticalTests` | 4 | Paired t-test, Wilcoxon, Welch's t-test, Mann-Whitney U all produce valid p-values |
| `TestCalculateRegimeProbability` | 2 | Returns (float, Series, float) tuple, probability in [0, 100] |
| `TestGetTickerStyle` | 4 | Winner/loser/benchmark/unknown ticker styling with correct colors, dash patterns, widths |

### Test Import Strategy

Tests import `app.py` functions without triggering Streamlit UI code using a comprehensive mock:

- **`_DummyContext`** — catch-all class that acts as context manager, callable, and attribute sink for all Streamlit widgets
- **`_SessionState(dict)`** — dict subclass supporting attribute-style access for `st.session_state`
- **`st.stop()` → `SystemExit`** — module-level UI code halts cleanly after all function definitions are loaded
- **Mock yfinance** — prevents network calls during test collection

### Run Locally

```bash
# Run all unit tests
pytest tests/ -v

# Run manual validation (imports, syntax, data sources, API key)
python validate.py

# Syntax check only
python -m py_compile app.py
```

---

### V — Validate

#### Quantitative Validation

| Metric | Expected | Observed | Status |
|--------|----------|----------|--------|
| NVDA outperforms SPY | Yes | +800%+ vs ~40% | Confirmed |
| CHGG negative returns | Yes | -90%+ decline | Confirmed |
| Entropy decreases post-launch | Yes | Winner concentration increasing | Confirmed |
| Regime breaks detected at key dates | Yes | Multiple CUSUM breaks found | Confirmed |
| Winner/Loser divergence statistically significant | Yes | p < 0.05 (Welch's t-test, Mann-Whitney U) | Confirmed |
| AI sector market cap expanded | Yes | 3x+ growth multiple | Confirmed |
| 21/21 unit tests pass | Yes | All green in CI | Confirmed |

#### Hypothesis Testing Results

**H1: Creative Destruction** — CONFIRMED
- Chegg lost 49% in a single day (May 2, 2023); CEO explicitly cited ChatGPT
- Total decline exceeds 90% from pre-ChatGPT levels
- All 5 disrupted education stocks show significant negative returns
- Paired t-tests confirm statistically significant return deterioration per ticker

**H2: Sample Space Expansion** — CONFIRMED
- "AI infrastructure" emerged as a distinct investment theme
- NVDA market cap grew from ~$400B to ~$3T+
- AI sector market cap crossed $1T, $5T, and $10T thresholds in sequence
- Historical comparison shows growth multiples exceeding dot-com and mobile regime shifts

**H3: Winner-Take-Most** — CONFIRMED
- NVDA dominates AI winners basket (stock-level entropy declining)
- Sector-level entropy reveals AI Infrastructure capturing disproportionate share
- Welch's t-test and Mann-Whitney U confirm statistically significant Winner/Loser group divergence
- Concentration index increased post-launch across all rolling windows

#### Visual Validation

The dashboard provides visual confirmation across 8 tabs:
1. **Divergence Chart**: Clear separation between winners and losers post-launch
2. **Returns Bar**: Magnitude of outperformance visible at a glance
3. **Entropy Plots**: Stock-level and sector-level declining entropy trends
4. **Regime Chart**: Structural breaks identified at ChatGPT launch and NVIDIA earnings
5. **Statistical Tables**: p-values and significance indicators for all inferential tests
6. **Sample Space Charts**: Market cap threshold crossings annotated on timeline

---

### E — Evolve

#### Project Evolution: Project 1 → Project 2 → Final Project

| Milestone | Date | Key Additions | Feedback Addressed |
|-----------|------|---------------|-------------------|
| **Project 1** | Feb 4, 2026 | Initial 6-tab dashboard with 6 tickers, Shannon entropy, CUSUM regime detection, pre/post statistical comparison | *Initial submission* |
| **Project 2** | Feb 18, 2026 | Fixed entropy (market cap), fixed Sharpe ratio (^IRX), regime prediction tab, historical CUSUM validation, early warning system, portfolio simulator, AI Market Analyst tab | "Entropy should use market cap not price", "Sharpe ratio needs risk-free rate", "Add forward-looking analysis" |
| **Final Project** | Feb 28, 2026 | 15 tickers, inferential tests, sector entropy, sample space expansion, CI/CD pipeline, simplified slider | "Expand ticker universe", "Add statistical tests", "Use sector-level entropy", "Measure sample space directly", "Simplify slider" |

#### Project 1 → Project 2 Improvements

- [x] Fixed entropy calculation — actual market capitalization (price × shares outstanding) instead of price proxy
- [x] Fixed Sharpe ratio — subtracts live 3-month Treasury yield (^IRX) instead of raw return/volatility
- [x] Added regime prediction — forward-looking CUSUM percentile ranking with probability gauge
- [x] Added historical validation — CUSUM tested against iPhone launch, Bitcoin surge, COVID crash, ChatGPT launch
- [x] Added early warning system — entropy trend, correlation structure shift, volatility regime monitoring
- [x] Added portfolio simulator — $10K investment comparison across AI Winners, SPY, and Disrupted Basket
- [x] Added AI Market Analyst — GPT-4o-powered contextual analysis with conversational follow-up via OpenRouter

#### Project 2 → Final Project Improvements

- [x] **Expanded ticker universe** — 6 to 15 stocks: 9 AI winners (added AMD, AVGO, ORCL, CRM, PLTR) and 5 disrupted companies (added PRSO, TAL, UDMY, COUR)
- [x] **Added inferential statistical tests** — paired t-test, Wilcoxon signed-rank, Welch's two-sample t-test, and Mann-Whitney U confirming statistically significant Winner/Loser divergence
- [x] **Added sector-level entropy** — stocks grouped into 4 sectors with stock vs sector entropy comparison, stacked area chart, and treemap of current sector weights
- [x] **Added sample space expansion analysis** — AI sector market cap growth multiple, threshold crossing timeline, and historical comparison to dot-com and mobile regime shifts
- [x] **Added CI/CD pipeline** — GitHub Actions workflow with syntax check, 21 pytest unit tests, and flake8 linting on every push/PR
- [x] **Simplified portfolio date slider** — replaced complex slider + buttons with clean 5-option select slider (professor feedback)

#### Remaining Limitations

1. **Universe Scope**: 15 tickers across 4 sectors — could expand to content creation, advertising, healthcare AI
2. **No Real-Time Streaming**: Data refreshes on page load (1-hour cache)
3. **Single LLM Provider**: Currently GPT-4o via OpenRouter; could add model selection

#### Future Enhancements

- [ ] Options market implied volatility analysis
- [ ] Sentiment analysis from earnings call transcripts
- [ ] Factor decomposition (isolate AI factor from market beta)
- [ ] Cross-asset analysis (bonds, commodities, crypto)
- [ ] International markets comparison (ASML, TSM, SAP)

#### Extensibility

The codebase is designed for easy extension:
```python
# Add new tickers — just append to the list
WINNERS = ['NVDA', 'MSFT', 'META', 'GOOGL', 'AMD', 'AVGO', 'ORCL', 'CRM', 'PLTR']
LOSERS = ['CHGG', 'PRSO', 'TAL', 'UDMY', 'COUR']

# Add new sectors — add entry to SECTORS dict
SECTORS = {
    'AI Infrastructure': ['NVDA', 'AMD', 'AVGO'],
    'Cloud/Enterprise AI': ['MSFT', 'GOOGL', 'ORCL', 'CRM', 'PLTR'],
    'Social/Consumer AI': ['META'],
    'Education (Disrupted)': ['CHGG', 'PRSO', 'TAL', 'UDMY', 'COUR'],
}

# Colors auto-resolve via TICKER_COLORS dict and get_ticker_style()
```

---

### R — Reflect

#### Technical Learnings

1. **yfinance Reliability**: Generally reliable but occasional timeout issues
   - Solution: `@st.cache_data` with 1-hour TTL and graceful error handling

2. **Streamlit Performance**: 15 tickers with multiple chart types can slow UI
   - Solution: Aggressive caching, lazy computation inside tab blocks

3. **Sharpe Ratio Accuracy**: Hardcoded risk-free rates introduce silent errors
   - Solution: Live 3-month Treasury yield from ^IRX with 4.5% fallback

4. **Market Cap vs Price for Entropy**: Price-based weights distort concentration metrics for stocks with vastly different share counts
   - Solution: Fetch shares outstanding via `yfinance.Ticker.info` and compute actual market cap

5. **Streamlit Session State**: Essential for maintaining LLM conversation history across reruns
   - `st.chat_input` has tab-pinning issues; `st.text_input` + button is more reliable

6. **OpenRouter as LLM Gateway**: The OpenAI SDK works with any OpenAI-compatible endpoint via `base_url`
   - Avoids vendor lock-in while keeping the same code structure

7. **Testing Streamlit Apps**: Module-level UI code makes direct imports fail
   - Solution: Comprehensive mock with `_DummyContext` catch-all, `_SessionState(dict)`, and `st.stop()` → `SystemExit` to halt after function definitions load

#### Methodological Insights

1. **Regime Detection Sensitivity**: CUSUM threshold selection is critical
   - Lower threshold = more breaks detected; higher = only major breaks
   - Chose 2.0-2.5 as balance between sensitivity and specificity

2. **Entropy at Two Levels**: Stock-level entropy alone can be misleading
   - Sector-level entropy adds a higher-order view of where concentration happens
   - Comparing both reveals whether value is concentrating *between* or *within* sectors

3. **Historical Validation Builds Confidence**: CUSUM successfully flags known events (COVID crash, ChatGPT launch) within days, lending credibility to forward-looking predictions

4. **LLM + Structured Data**: Feeding pre-computed dashboard metrics (returns, CUSUM, entropy, warnings) as context produces more grounded analysis than open-ended prompting

5. **Inferential Tests Complement Visual Analysis**: Charts show the *what*; t-tests and Wilcoxon tests prove it's not noise (p < 0.05)

#### Market Understanding Gained

1. **Speed of Repricing**: Markets repriced entire value chains within months of ChatGPT's release
2. **Second-Order Effects**: Not just OpenAI benefited — the entire GPU, cloud, and enterprise AI infrastructure ecosystem saw massive repricing
3. **Non-Linear Impacts**: Losers didn't just underperform; several faced existential threats (Chegg -90%+, Pearson restructuring)
4. **Sample Space Expansion is Measurable**: The AI sector's market cap growth from ~$3.8T to ~$13T+ provides concrete evidence of new value creation, not just redistribution

#### Framework Effectiveness

The DRIVER framework proved valuable for:
- **Structured iteration**: Each project milestone built on the previous (Project 2 → Final Project)
- **Clear documentation**: README doubles as methodology documentation and project report
- **Reproducible analysis**: Automated tests ensure calculations remain correct as features are added
- **Stakeholder communication**: Dashboard tabs map directly to DRIVER phases

---

## Academic Context

### Course Information
- **Course**: MGMT 69000 — Mastering AI for Finance
- **Institution**: Purdue University, Daniels School of Business
- **Submission**: Final Project
- **Topic**: Regime Shifts and Sample Space Expansion in Financial Markets

### Key Concepts Demonstrated

1. **Regime Shifts**: Fundamental, statistically detectable changes in market dynamics (CUSUM, inferential tests)
2. **Creative Destruction**: Schumpeterian disruption of existing business models (education sector collapse)
3. **Sample Space Expansion**: Emergence of new asset classes and return distributions (AI infrastructure market cap growth)
4. **Winner-Take-Most**: Concentration dynamics in technology markets (declining Shannon entropy)

### Further Reading

- Sornette, D. (2003). *Why Stock Markets Crash: Critical Events in Complex Financial Systems*
- Taleb, N.N. (2007). *The Black Swan: The Impact of the Highly Improbable*
- Arthur, W.B. (2009). *The Nature of Technology: What It Is and How It Evolves*
- Schumpeter, J.A. (1942). *Capitalism, Socialism and Democracy* — creative destruction theory

---

## License

MIT License — See LICENSE file for details.

## Acknowledgments

- OpenAI for ChatGPT (the catalyst studied) and GPT-4o (the analyst powering tab 8)
- [OpenRouter](https://openrouter.ai/) for LLM API routing
- Purdue University MGMT 69000 course and professor feedback that shaped the final project
- Yahoo Finance for market data
- Streamlit, Plotly, scipy, and the Python open-source ecosystem

---

*Built with the DRIVER methodology for quantitative finance research and validated by automated CI/CD.*
