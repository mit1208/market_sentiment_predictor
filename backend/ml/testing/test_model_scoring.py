from backend.ml.shared.model_scoring import blend_positive_class_scores, classify_probability


def test_blend_positive_class_scores_honors_metadata_weights():
    metadata = {
        "prediction_strategy": "weighted_ensemble_lgb_85",
        "ensemble_lgb_weight": 0.85,
        "classification_threshold": 0.61,
    }

    probability = blend_positive_class_scores(0.7, 0.2, metadata=metadata)

    assert round(probability, 4) == 0.625


def test_classify_probability_honors_metadata_threshold():
    metadata = {
        "prediction_strategy": "lightgbm_only",
        "ensemble_lgb_weight": 1.0,
        "classification_threshold": 0.61,
    }

    label, confidence, threshold = classify_probability(0.6, metadata=metadata)

    assert label == "Down"
    assert round(confidence, 4) == 0.4
    assert threshold == 0.61
