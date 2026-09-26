"""Tests for src/risk_scoring.py"""
from src.risk_scoring import probability_to_score, score_to_level, calculate_risk


def test_probability_to_score_bounds():
    assert probability_to_score(0.0) == 0
    assert probability_to_score(1.0) == 100
    assert probability_to_score(0.5) == 50


def test_probability_to_score_clamps_out_of_range():
    assert probability_to_score(-0.5) == 0
    assert probability_to_score(1.5) == 100


def test_score_to_level_boundaries():
    assert score_to_level(0) == "Low"
    assert score_to_level(30) == "Low"
    assert score_to_level(31) == "Medium"
    assert score_to_level(60) == "Medium"
    assert score_to_level(61) == "High"
    assert score_to_level(80) == "High"
    assert score_to_level(81) == "Critical"
    assert score_to_level(100) == "Critical"


def test_calculate_risk_output_shape():
    result = calculate_risk(0.87)
    assert set(result.keys()) == {"fraud_probability", "risk_score", "risk_level"}
    assert result["risk_score"] == 87
    assert result["risk_level"] == "Critical"
