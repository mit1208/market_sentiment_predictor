from backend.swarm.agents.moderator import moderate
from backend.swarm.models.schemas import AgentVerdict, MlPrediction


def test_moderator_keeps_model_direction_without_supermajority():
    verdicts = [
        AgentVerdict(
            deliberation_id="delib:AAPL:1",
            request_id="req:AAPL:1",
            ticker="AAPL",
            agent_id="sentiment_analyst",
            display_name="Sentiment Analyst",
            round=2,
            direction="Up",
            confidence_delta=0.05,
            conviction="medium",
            thesis="Supportive sentiment backdrop.",
            agrees_with_model=True,
            ml_direction="Up",
            ml_confidence=0.7,
        ),
        AgentVerdict(
            deliberation_id="delib:AAPL:1",
            request_id="req:AAPL:1",
            ticker="AAPL",
            agent_id="macro_economist",
            display_name="Macro Economist",
            round=2,
            direction="Down",
            confidence_delta=-0.06,
            conviction="high",
            thesis="Macro tape is soft.",
            agrees_with_model=False,
            ml_direction="Up",
            ml_confidence=0.7,
        ),
    ]
    summary = moderate(
        deliberation_id="delib:AAPL:1",
        request_id="req:AAPL:1",
        ticker="AAPL",
        ml_prediction=MlPrediction(direction="Up", confidence=0.7, probability=0.7),
        agent_verdicts=verdicts,
    )

    assert summary.final_direction == "Up"
    assert summary.direction_overridden is False
    assert summary.consensus == "3/4 agree"


def test_moderator_overrides_on_high_conviction_supermajority():
    verdicts = []
    for idx, agent_id in enumerate(
        ["sentiment_analyst", "macro_economist", "risk_manager", "contrarian"]
    ):
        verdicts.append(
            AgentVerdict(
                deliberation_id="delib:AAPL:2",
                request_id="req:AAPL:2",
                ticker="AAPL",
                agent_id=agent_id,
                display_name=agent_id,
                round=2,
                direction="Down" if idx < 3 else "Up",
                confidence_delta=-0.12 if idx < 3 else 0.01,
                conviction="high" if idx < 2 else "medium",
                thesis="Stress signal.",
                agrees_with_model=False if idx < 3 else True,
                ml_direction="Up",
                ml_confidence=0.72,
            )
        )
    summary = moderate(
        deliberation_id="delib:AAPL:2",
        request_id="req:AAPL:2",
        ticker="AAPL",
        ml_prediction=MlPrediction(direction="Up", confidence=0.72, probability=0.72),
        agent_verdicts=verdicts,
    )

    assert summary.final_direction == "Down"
    assert summary.direction_overridden is True
    assert 0.30 <= summary.final_confidence <= 0.95
