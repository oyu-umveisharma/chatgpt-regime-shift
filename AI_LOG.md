# AI Collaboration Log

**MGMT 69000: Mastering AI for Finance | Purdue University | Final Project**

---

## Project Overview

- **Tool**: Claude Code with DRIVER plugin
- **Model**: Claude (Anthropic Opus 4.6)
- **Total Sessions**: 3 (Feb 4, Feb 18, Feb 28)
- **Repository**: [github.com/oyu-umveisharma/chatgpt-regime-shift](https://github.com/oyu-umveisharma/chatgpt-regime-shift)

---

## Session 1: Project 1 (Feb 4, 2026)

### Prompts Used

- "Build a Streamlit dashboard analyzing the ChatGPT Launch as a regime shift event in financial markets"
- "Add normalized price chart rebased to 100 at ChatGPT launch date"
- "Add Shannon entropy calculation for market concentration"
- "Add CUSUM regime detection for structural break analysis"
- "Fix timezone error in normalize_prices — Cannot compare dtypes datetime64[ns, America/New_York] and datetime64[ns]"
- "Add Deep Analysis tab with pre/post statistical comparison and correlation heatmaps"
- "Add Chegg crash deep-dive chart with event annotations"

### What AI Generated

- Initial `app.py` structure with 6 tabs (Price Performance, Total Returns, Market Concentration, Regime Detection, Deep Analysis, Portfolio Impact)
- Data fetching via yfinance with `@st.cache_data` caching
- Shannon entropy calculation using price as weight proxy
- CUSUM regime detection with `scipy.signal.find_peaks`
- Normalized price chart, returns bar chart, entropy chart, regime chart
- Pre/post statistical comparison table with Sharpe ratios
- Correlation heatmaps (pre and post ChatGPT launch)
- Chegg crash deep-dive with volume spike and event annotations

### What I Modified

- Debugged timezone mismatch by adding `df.index = df.index.tz_localize(None)` to `fetch_stock_data()`
- Adjusted chart colors and layout for presentation clarity
- Added personal interpretations and analysis narratives to each tab
- Selected the initial 6 tickers: NVDA, MSFT, META, GOOGL, CHGG, SPY

---

## Session 2: Project 2 (Feb 18, 2026)

### Prompts Used

- "Fix Sharpe ratio calculation to subtract the risk-free rate instead of using raw return/volatility"
- "Fix Shannon entropy to use actual market cap (price x shares outstanding) instead of price proxy"
- "Add Regime Prediction tab with forward-looking CUSUM percentile ranking"
- "Validate CUSUM detector against historical events: iPhone launch, Bitcoin surge, COVID crash"
- "Add early warning system with entropy trend, correlation shift, and volatility regime signals"
- "Add portfolio simulator comparing $10K in AI Winners vs SPY vs CHGG"
- "Add AI Market Analyst tab using Anthropic Claude API"
- "Switch the AI tab from Anthropic to OpenAI GPT API"
- "Update the AI tab to use OpenRouter API instead of direct OpenAI"

### What AI Generated

- `fetch_risk_free_rate()` function fetching 3-month Treasury yield from ^IRX via yfinance
- Corrected Sharpe ratio: `(R_annualized - R_f) / σ_annualized`
- Market cap calculation using `yfinance.Ticker.info['sharesOutstanding']` for entropy weights
- `calculate_regime_probability()` using CUSUM percentile ranking against historical distribution
- `validate_cusum_on_event()` for testing CUSUM on known historical regime shifts
- `assess_early_warnings()` returning entropy trend, correlation shift, and volatility regime signals
- `simulate_portfolio()` comparing three strategies with adjustable start dates
- `call_openai_analyst()` with OpenRouter routing via OpenAI SDK `base_url`
- `AI_ANALYST_SYSTEM_PROMPT` grounding GPT-4o in the regime shift thesis
- `st.session_state.ai_chat_history` for conversational persistence

### What I Modified

- Chose OpenRouter over direct Anthropic/OpenAI after testing revealed API key incompatibilities
- Tested and validated all regime prediction outputs against known events
- Verified early warning signals produce meaningful results with live data
- Wrote presentation script and talking points for class demo
- Selected `st.text_input` + button over `st.chat_input` due to Streamlit tab-pinning issues

---

## Session 3: Final Project (Feb 28, 2026)

### Prompts Used

- "Leverage driver plugin financial application builder. Expand the ticker universe from 6 stocks to 15 stocks. This addresses professor feedback: 'Expand ticker universe beyond 6 stocks.'"
- "Leverage driver plugin financial application builder. Add inferential statistical tests to the Deep Analysis tab. This addresses professor feedback: 'Add inferential statistical tests (t-test, Wilcoxon).'"
- "Leverage driver plugin financial application builder. Add sector-level entropy analysis to the Market Concentration tab. This addresses professor feedback: 'Use sector-level data for more meaningful entropy.'"
- "Leverage driver plugin financial application builder. Add direct measurement of sample space expansion to the Deep Analysis tab. This addresses professor feedback: 'Measure sample space expansion hypothesis directly.'"
- "Leverage driver plugin financial application builder. Add CI/CD pipeline using GitHub Actions."
- "Leverage driver plugin financial application builder. Simplify the portfolio chart date slider. This addresses professor feedback: 'portfolio chart date slider could be simplified.'"

### What AI Generated

**Ticker Expansion:**
- Extended `WINNERS` (4 → 9), `LOSERS` (1 → 5), total 6 → 15 tickers
- `TICKER_COLORS` dict with distinct colors per ticker
- `get_ticker_style()` function returning `(color, dash, width)` tuples
- Refactored all chart functions to use centralized style system
- Equal-weight "Disrupted Basket" portfolio replacing single CHGG

**Statistical Tests:**
- Per-ticker paired t-test and Wilcoxon signed-rank test (pre vs post 60-day returns)
- Winners vs Losers Welch's t-test (`scipy.stats.ttest_ind`, `equal_var=False`)
- Winners vs Losers Mann-Whitney U test (`scipy.stats.mannwhitneyu`)
- Styled results tables with significance indicators and basis point formatting

**Sector Entropy:**
- `SECTORS` dict grouping 14 tickers into 4 sectors
- `SECTOR_COLORS` dict for consistent visualization
- `calculate_sector_entropy()` function with rolling window
- Stock-level vs sector-level entropy comparison chart
- Stacked area chart of sector market cap shares
- Treemap of current sector weights

**Sample Space Expansion:**
- Pre-launch vs current market cap comparison with growth multiples
- Threshold crossing analysis ($100B, $500B, $1T, $5T, $10T)
- Stacked area chart with threshold annotations
- Historical comparison table (dot-com, mobile, AI)

**CI/CD Pipeline:**
- `.github/workflows/ci.yml` — GitHub Actions with syntax check, pytest, flake8
- `tests/test_app.py` — 21 unit tests across 7 test classes
- `_DummyContext` mock for Streamlit, `_SessionState(dict)` for session state
- `validate.py` — manual validation script
- CI badge in README

**Slider Simplification:**
- Replaced `st.slider` + 4 buttons (28 lines) with `st.select_slider` + 5 presets (13 lines)

### What I Modified

- Selected which 9 new stocks to add based on sector representation and market relevance
- Defined the 4 sector groupings (AI Infrastructure, Cloud/Enterprise, Social/Consumer, Education)
- Chose the specific statistical tests to match professor's requirement for both parametric and non-parametric
- Validated that group-level tests produce p < 0.05 with real market data
- Selected the 5 preset dates for the simplified portfolio slider
- Reviewed CI/CD configuration and resolved PAT `workflow` scope issue
- Debugged test failures: fixed `_DummyContext` to return instances instead of `None`, fixed `_SessionState` for attribute-style access, tuned test parameters for reliable p-value assertions

---

## Ownership Verification

For each component of this dashboard, I can:

1. **Explain what it does** — I understand the purpose of every function, from Shannon entropy calculation to CUSUM regime detection to OpenRouter API routing
2. **Explain why this approach** — I chose paired t-tests for within-ticker comparison and Welch's t-test for group comparison because financial returns violate equal-variance assumptions; I chose sector-level entropy because stock-level entropy alone doesn't distinguish between-sector vs within-sector concentration
3. **Modify if requirements changed** — I could add new tickers by appending to `WINNERS`/`LOSERS` and `TICKER_COLORS`; I could add a new sector by extending the `SECTORS` dict; I could swap GPT-4o for another model by changing one line in `call_openai_analyst()`
4. **Catch errors if wrong** — I identified and fixed the CI/CD test failures (mock import strategy), the timezone mismatch, the Sharpe ratio bug, and the entropy calculation error — each required understanding the root cause, not just the symptom

---

## Key Learnings

### On AI-Assisted Development

- **DRIVER methodology structures AI collaboration** — each prompt maps to a DRIVER phase (Discover → research question, Represent → data structures, Implement → code, Validate → testing, Evolve → iteration, Reflect → documentation)
- **AI accelerates implementation, human drives decisions** — the AI wrote the scipy integration, but I chose which tests to run and interpreted the p-values; the AI built the sector groupings, but I decided which companies belong in which sector
- **Structured prompts produce structured code** — numbered requirements lists (e.g., "1. Add paired t-test, 2. Add Wilcoxon, 3. Add results table") consistently generated well-organized implementations with minimal revision
- **Professor feedback as prompt input** — treating rubric feedback as requirements specifications created a direct pipeline from classroom evaluation to code improvement

### On Validation

- **Automated testing catches silent failures** — the CI/CD pipeline revealed 18/21 test failures that manual dashboard testing missed (the app ran fine, but the test import strategy was broken)
- **Mock engineering is non-trivial** — the hardest technical challenge was not the financial analysis — it was getting pytest to import a Streamlit app without triggering 1200+ lines of UI code
- **Incremental commits preserve progress** — each feature was committed separately, creating a clean git history that documents the evolution from Project 1 to Final Project

### On Financial Analysis

- **Statistical significance matters** — visual charts show divergence, but inferential tests (p < 0.05) prove it's not noise
- **Multi-level entropy reveals more** — stock-level entropy shows NVDA dominance, but sector-level entropy shows AI Infrastructure as a whole is capturing disproportionate value
- **Sample space expansion is measurable** — the AI sector's 3.4x market cap growth provides concrete evidence of new value creation, comparable to dot-com and mobile regime shifts

---

*This log was created as a required deliverable for the MGMT 69000 Final Project to document AI-assisted development practices and demonstrate understanding of all generated code.*
