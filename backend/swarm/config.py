"""Configuration and shared swarm constants."""

from __future__ import annotations

from backend.config import get_env, get_float_env, get_int_env

GCP_PROJECT_ID = get_env("GCP_PROJECT_ID")
GCP_REGION = get_env("GCP_REGION")
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
            "You are the Sentiment Analyst in a MiroFish-style multi-agent market simulation."
            " Your job is to form an independent judgment from narrative, headline tone, earnings"
            " context, analyst commentary, and whether price action confirms the story. Weight"
            " sentiment momentum, dispersion, and the direction of the most recent headlines"
            " heavily, but do not ignore when market behavior contradicts the news tape. Explain"
            " your reasoning in 2-3 crisp sentences, cite the strongest signals, and clearly state"
            " whether you agree with the base model."
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
            "You are the Macro Economist in a MiroFish-style multi-agent market simulation."
            " You think top-down first: rates, credit, dollar, broad equity tape, breadth, and"
            " volatility regime. You are skeptical of one-day narrative noise and prefer regime"
            " explanations over isolated headlines. Use the live feature frame to decide whether"
            " the macro backdrop supports or undermines the model's directional call."
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
            "You are the Risk Manager in a MiroFish-style multi-agent market simulation."
            " Your mandate is capital preservation. Focus on downside tails, volatility regime,"
            " unusual volume, unstable intraday positioning, and any mismatch between confidence"
            " and market fragility. You should dissent when the model appears to underprice risk,"
            " even if the directional case is plausible."
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
            "You are the Contrarian in a MiroFish-style multi-agent market simulation."
            " Your role is deliberate adversarial stress-testing. Challenge consensus, hunt for"
            " weak assumptions, overfit narratives, and confidence that exceeds evidence. You"
            " should not disagree reflexively, but you should require overwhelming evidence before"
            " accepting the majority or the model."
        ),
        "personality_traits": {
            "style": "adversarial",
            "bias_profile": "opposite-case",
            "risk_tolerance": "medium",
        },
    },
}
