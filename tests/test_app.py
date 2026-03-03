"""Unit tests for ChatGPT Regime Shift Dashboard core functions."""

import pytest
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_price_data():
    """Create sample price data for two tickers over ~120 trading days."""
    dates = pd.bdate_range("2022-06-01", periods=120)
    np.random.seed(42)

    data = {}
    for ticker, start_price, drift in [("NVDA", 150, 0.003), ("CHGG", 30, -0.002)]:
        returns = np.random.normal(drift, 0.02, len(dates))
        prices = start_price * np.cumprod(1 + returns)
        df = pd.DataFrame({"Close": prices}, index=dates)
        df["SharesOutstanding"] = 1_000_000_000
        df["MarketCap"] = df["Close"] * df["SharesOutstanding"]
        data[ticker] = df

    return data


@pytest.fixture
def sample_returns_series():
    """Create a sample daily returns series with a known regime break."""
    np.random.seed(42)
    # Regime 1: low mean
    r1 = np.random.normal(0.0005, 0.01, 100)
    # Regime 2: high mean (simulates break)
    r2 = np.random.normal(0.005, 0.02, 100)
    returns = np.concatenate([r1, r2])
    dates = pd.bdate_range("2022-06-01", periods=len(returns))
    return pd.Series(returns, index=dates)


@pytest.fixture
def base_date():
    return pd.Timestamp("2022-08-01")


# ---------------------------------------------------------------------------
# Import helpers — import functions without triggering Streamlit
# ---------------------------------------------------------------------------

class _DummyContext:
    """Catch-all mock that acts as context manager, callable, and attribute sink."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def __call__(self, *a, **kw):
        return _DummyContext()
    def __getattr__(self, name):
        return _DummyContext()


class _SessionState(dict):
    """Dict subclass that supports attribute-style access (like st.session_state)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def _stop():
    """Mock st.stop() — raises SystemExit to halt module-level UI code."""
    raise SystemExit("st.stop")


def _import_app_functions():
    """Import app.py functions by patching Streamlit to avoid UI side effects."""
    import importlib
    import sys

    # Build a comprehensive Streamlit mock.
    # Use _DummyContext as the base — any un-mocked attribute returns a no-op.
    mock_st = _DummyContext()

    # Override the attributes that need specific behavior:
    mock_st.cache_data = lambda **kw: (lambda f: f)
    mock_st.set_page_config = lambda **kw: None
    mock_st.session_state = _SessionState()
    mock_st.sidebar = _DummyContext()
    mock_st.stop = _stop
    mock_st.columns = lambda n, **kw: [_DummyContext() for _ in range(n)]
    mock_st.tabs = lambda labels, **kw: [_DummyContext() for _ in labels]
    mock_st.button = lambda *a, **kw: False
    mock_st.text_input = lambda *a, **kw: ""
    mock_st.slider = lambda *a, **kw: 0

    sys.modules["streamlit"] = mock_st

    # Also mock yfinance to avoid network calls during import
    mock_yf = _DummyContext()
    mock_yf.Ticker = lambda t: type(
        "T", (), {
            "history": lambda self=None, **kw: pd.DataFrame(),
            "info": {},
        }
    )()
    sys.modules["yfinance"] = mock_yf

    # Import the module
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app = importlib.util.module_from_spec(spec)

    # Execute module — UI code hits st.stop() → SystemExit, which is expected.
    # All function defs execute before that point.
    try:
        spec.loader.exec_module(app)
    except SystemExit:
        pass  # Expected: st.stop() fires because mock data is empty

    return app


# Module-level import (runs once)
_app = _import_app_functions()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchRiskFreeRate:
    def test_returns_float(self):
        """fetch_risk_free_rate should return a float."""
        result = _app.fetch_risk_free_rate()
        assert isinstance(result, float)

    def test_returns_reasonable_value(self):
        """Risk-free rate should be between 0 and 20%."""
        result = _app.fetch_risk_free_rate()
        assert 0 < result < 20


class TestCalculateReturns:
    def test_returns_dict(self, sample_price_data, base_date):
        """calculate_returns should return a dict of ticker -> return %."""
        result = _app.calculate_returns(sample_price_data, base_date)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "NVDA" in result
        assert "CHGG" in result

    def test_returns_are_numeric(self, sample_price_data, base_date):
        """Each return value should be a float."""
        result = _app.calculate_returns(sample_price_data, base_date)
        for ticker, ret in result.items():
            assert isinstance(ret, (int, float, np.floating)), f"{ticker} return is {type(ret)}"

    def test_positive_drift_positive_return(self, sample_price_data, base_date):
        """NVDA (positive drift) should have positive return over the period."""
        result = _app.calculate_returns(sample_price_data, base_date)
        assert result["NVDA"] > 0


class TestCalculateRollingEntropy:
    def test_returns_series(self, sample_price_data):
        """calculate_rolling_entropy should return a Series and fallback list."""
        entropy, fallback = _app.calculate_rolling_entropy(sample_price_data, window=30)
        assert isinstance(entropy, pd.Series)
        assert isinstance(fallback, list)

    def test_entropy_bounds(self, sample_price_data):
        """Entropy values should be between 0 and log2(n)."""
        entropy, _ = _app.calculate_rolling_entropy(sample_price_data, window=30)
        clean = entropy.dropna()
        if len(clean) > 0:
            n_stocks = 2  # NVDA and CHGG
            max_entropy = np.log2(n_stocks)
            assert clean.min() >= 0
            assert clean.max() <= max_entropy + 0.01  # small float tolerance


class TestDetectRegimeBreaks:
    def test_returns_list(self, sample_returns_series):
        """detect_regime_breaks should return a list."""
        breaks = _app.detect_regime_breaks(sample_returns_series, threshold=2.0)
        assert isinstance(breaks, list)

    def test_detects_known_break(self, sample_returns_series):
        """Should detect at least one break in data with an obvious regime change."""
        breaks = _app.detect_regime_breaks(sample_returns_series, threshold=1.5)
        assert len(breaks) >= 1

    def test_break_dates_are_timestamps(self, sample_returns_series):
        """Break dates should be Timestamp objects."""
        breaks = _app.detect_regime_breaks(sample_returns_series, threshold=1.5)
        for b in breaks:
            assert isinstance(b, pd.Timestamp)

    def test_empty_with_short_series(self):
        """Should return empty list for very short series."""
        short = pd.Series([0.01, -0.01, 0.02], index=pd.bdate_range("2022-01-01", periods=3))
        breaks = _app.detect_regime_breaks(short)
        assert breaks == []


class TestStatisticalTests:
    def test_ttest_returns_pvalue(self):
        """scipy ttest_rel should return valid t-stat and p-value."""
        np.random.seed(42)
        pre = np.random.normal(0.001, 0.02, 100)
        post = np.random.normal(0.003, 0.02, 100)
        t_stat, p_val = stats.ttest_rel(post, pre)
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1

    def test_wilcoxon_returns_pvalue(self):
        """scipy wilcoxon should return valid stat and p-value."""
        np.random.seed(42)
        pre = np.random.normal(0.001, 0.02, 100)
        post = np.random.normal(0.003, 0.02, 100)
        w_stat, p_val = stats.wilcoxon(post - pre)
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1

    def test_ttest_ind_winners_vs_losers(self):
        """Two-sample t-test should work for winner vs loser groups."""
        np.random.seed(42)
        winners = np.random.normal(0.005, 0.02, 500)
        losers = np.random.normal(-0.003, 0.02, 500)
        t_stat, p_val = stats.ttest_ind(winners, losers, equal_var=False)
        assert isinstance(p_val, float)
        assert p_val < 0.05  # should be significant with these parameters

    def test_mannwhitneyu(self):
        """Mann-Whitney U should return valid stat and p-value."""
        np.random.seed(42)
        winners = np.random.normal(0.003, 0.02, 200)
        losers = np.random.normal(-0.001, 0.02, 200)
        u_stat, p_val = stats.mannwhitneyu(winners, losers, alternative="two-sided")
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1


class TestCalculateRegimeProbability:
    def test_returns_tuple(self, sample_returns_series):
        """calculate_regime_probability should return (prob, series, value)."""
        recent = sample_returns_series.iloc[-60:]
        full = sample_returns_series
        prob, cusum, val = _app.calculate_regime_probability(recent, full)
        assert isinstance(prob, float)
        assert isinstance(cusum, pd.Series)
        assert isinstance(val, float)

    def test_probability_range(self, sample_returns_series):
        """Probability should be between 0 and 100."""
        recent = sample_returns_series.iloc[-60:]
        full = sample_returns_series
        prob, _, _ = _app.calculate_regime_probability(recent, full)
        assert 0 <= prob <= 100


class TestGetTickerStyle:
    def test_winner_style(self):
        color, dash, width = _app.get_ticker_style("NVDA")
        assert dash == "solid"
        assert width == 2

    def test_loser_style(self):
        color, dash, width = _app.get_ticker_style("CHGG")
        assert dash == "dash"

    def test_benchmark_style(self):
        color, dash, width = _app.get_ticker_style("SPY")
        assert dash == "dot"

    def test_unknown_ticker(self):
        color, dash, width = _app.get_ticker_style("UNKNOWN")
        assert color == "#333333"
