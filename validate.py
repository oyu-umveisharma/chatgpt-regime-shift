"""
Validation script for ChatGPT Regime Shift Dashboard.
Run manually: python validate.py
"""

import sys


def check_imports():
    """Verify all required packages are importable."""
    required = [
        "streamlit", "yfinance", "pandas", "numpy",
        "plotly", "scipy", "openai", "dotenv",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_syntax():
    """Verify app.py has valid Python syntax."""
    import ast
    try:
        with open("app.py") as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_data_sources():
    """Verify Yahoo Finance is accessible for key tickers."""
    import yfinance as yf
    test_tickers = ["NVDA", "SPY", "CHGG"]
    results = {}
    for ticker in test_tickers:
        try:
            df = yf.Ticker(ticker).history(period="5d")
            results[ticker] = len(df) > 0
        except Exception:
            results[ticker] = False
    return results


def check_api_key():
    """Check if OpenRouter API key is configured."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return "missing"
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("sk-"):
        return "openai"
    return "unknown"


def main():
    print("=" * 60)
    print("ChatGPT Regime Shift Dashboard — Validation")
    print("=" * 60)
    all_ok = True

    # 1. Imports
    print("\n[1/4] Checking imports...")
    missing = check_imports()
    if missing:
        print(f"  FAIL: Missing packages: {', '.join(missing)}")
        all_ok = False
    else:
        print("  OK: All packages importable")

    # 2. Syntax
    print("\n[2/4] Checking syntax...")
    ok, err = check_syntax()
    if ok:
        print("  OK: app.py syntax valid")
    else:
        print(f"  FAIL: {err}")
        all_ok = False

    # 3. Data sources
    print("\n[3/4] Checking data sources...")
    results = check_data_sources()
    for ticker, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {status}: {ticker}")
    if not all(results.values()):
        all_ok = False

    # 4. API key
    print("\n[4/4] Checking API key...")
    key_status = check_api_key()
    if key_status == "missing":
        print("  WARN: No OPENAI_API_KEY in .env (AI tab will show error)")
    elif key_status == "openrouter":
        print("  OK: OpenRouter API key configured")
    elif key_status == "openai":
        print("  OK: OpenAI API key configured")
    else:
        print("  WARN: API key format not recognized")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: All checks passed")
    else:
        print("RESULT: Some checks failed (see above)")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
