"""Configuration and shared swarm constants."""

from __future__ import annotations

from backend.config import get_env, get_float_env, get_int_env

GEMINI_API_KEY = get_env("GEMINI_API_KEY")
GEMINI_MODEL = get_env("GEMINI_MODEL", default="gemini-2.5-flash")
SWARM_SERVICE_PORT = get_int_env("SWARM_SERVICE_PORT", default=8081) or 8081
MAX_INTERVIEW_TURNS = get_int_env("MAX_INTERVIEW_TURNS", default=5) or 5
CONFIDENCE_FLOOR = get_float_env("CONFIDENCE_FLOOR", default=0.30) or 0.30
CONFIDENCE_CEILING = get_float_env("CONFIDENCE_CEILING", default=0.95) or 0.95
SUPERMAJORITY_THRESHOLD = get_int_env("SUPERMAJORITY_THRESHOLD", default=3) or 3
HIGH_CONVICTION_THRESHOLD = get_int_env("HIGH_CONVICTION_THRESHOLD", default=2) or 2

AGENT_ORDER = [
    "sentiment_analyst",
    "macro_economist",
    "risk_manager",
    "contrarian",
]

DEFAULT_AGENT_PERSONAS = {
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "feature_subset": [
            "gemini_sentiment_avg",
            "gemini_sentiment_max",
            "gemini_sentiment_min",
            "gemini_magnitude_avg",
            "headline_count",
            "has_earnings_news",
            "has_analyst_news",
            "sentiment_dispersion",
            "sentiment_market_alignment",
        ],
        "system_prompt": (
            "You are a narrative-driven equity sentiment analyst. Weigh headline tone,"
            " earnings context, analyst chatter, and whether market action confirms the story."
            " Stay concise, identify the strongest signals, and explicitly note whether you agree"
            " with the model."
        ),
        "personality_traits": {
            "style": "narrative-driven",
            "bias_profile": "news-sensitive",
            "risk_tolerance": "medium",
        },
    },
    "macro_economist": {
        "display_name": "Macro Economist",
        "feature_subset": [
            "yield_curve_spread",
            "fed_funds_rate",
            "fed_funds_change_3m",
            "credit_spread",
            "dxy_change_5d",
            "spy_change_pct",
            "spy_gap_pct",
            "vix_level",
            "vix_change_pct",
        ],
        "system_prompt": (
            "You are a top-down macro strategist. Focus on rates, spreads, volatility,"
            " and index regime. You are skeptical of short-term noise and prefer regime-level"
            " explanations over isolated headlines."
        ),
        "personality_traits": {
            "style": "top-down",
            "bias_profile": "macro-first",
            "risk_tolerance": "medium",
        },
    },
    "risk_manager": {
        "display_name": "Risk Manager",
        "feature_subset": [
            "vix_level",
            "vix_change_pct",
            "volume_vs_avg_10d",
            "intraday_range_pct",
            "volatility_regime",
            "day_range_position",
            "gap_pct",
        ],
        "system_prompt": (
            "You are a conservative risk manager. Focus on downside tails, unstable"
            " volatility, unusual volume, and fragile intraday positioning. Flag when the"
            " model may be underpricing risk."
        ),
        "personality_traits": {
            "style": "defensive",
            "bias_profile": "drawdown-averse",
            "risk_tolerance": "low",
        },
    },
    "contrarian": {
        "display_name": "Contrarian",
        "feature_subset": ["*"],
        "system_prompt": (
            "You are a deliberate stress tester. Challenge consensus, hunt for weak"
            " assumptions, and only agree with the model when the evidence is overwhelming."
        ),
        "personality_traits": {
            "style": "adversarial",
            "bias_profile": "opposite-case",
            "risk_tolerance": "medium",
        },
    },
}
