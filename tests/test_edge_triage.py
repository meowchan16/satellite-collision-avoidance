"""
Automated Unit Tests for Onboard Edge AI Data Triage Engine.
Verifies real-time frame classification (cloud rejection, debris streak prioritization)
and downlink bandwidth savings calculations.
"""

import pytest
from core.edge_triage import OnboardSensorFrame, EdgeAITriageEngine, TriageDecision


@pytest.fixture
def triage_engine():
    """Provides a fresh EdgeAITriageEngine instance."""
    return EdgeAITriageEngine(cloud_threshold_pct=65.0, min_snr_threshold=2.0)


def test_cloudy_frame_discarded(triage_engine):
    """
    1. A cloudy frame (cloud_cover > 65%) must return DISCARD_OBSCURED_CLOUD
       and downlink 0 MB.
    """
    frame = OnboardSensorFrame(
        frame_id="FRAME_CLOUD_01",
        cloud_cover_pct=78.5,
        signal_to_noise_ratio=15.0,
        debris_streak_detected=False,
        raw_size_mb=40.0
    )

    decision = triage_engine.classify_frame(frame)
    assert decision == TriageDecision.DISCARD_OBSCURED_CLOUD.value
    assert triage_engine.total_raw_mb == 40.0
    assert triage_engine.total_downlinked_mb == 0.0


def test_debris_streak_priority_downlink(triage_engine):
    """
    2. A clear frame with debris_streak_detected=True must return PRIORITY_CONJUNCTION_HAZARD
       and downlink the full raw_size_mb.
    """
    frame = OnboardSensorFrame(
        frame_id="FRAME_DEBRIS_02",
        cloud_cover_pct=15.0,
        signal_to_noise_ratio=20.0,
        debris_streak_detected=True,
        raw_size_mb=35.0
    )

    decision = triage_engine.classify_frame(frame)
    assert decision == TriageDecision.PRIORITY_CONJUNCTION_HAZARD.value
    assert triage_engine.total_raw_mb == 35.0
    assert triage_engine.total_downlinked_mb == 35.0


def test_clear_frame_standard_downlink(triage_engine):
    """
    3. A clear frame with no debris detected must return STANDARD_DOWNLINK
       and downlink the full raw_size_mb.
    """
    frame = OnboardSensorFrame(
        frame_id="FRAME_CLEAR_03",
        cloud_cover_pct=25.0,
        signal_to_noise_ratio=18.0,
        debris_streak_detected=False,
        raw_size_mb=50.0
    )

    decision = triage_engine.classify_frame(frame)
    assert decision == TriageDecision.STANDARD_DOWNLINK.value
    assert triage_engine.total_raw_mb == 50.0
    assert triage_engine.total_downlinked_mb == 50.0


def test_low_snr_blank_frame_discarded(triage_engine):
    """
    Frames with SNR < 2.0 (blank or overexposed) must also be discarded.
    """
    frame = OnboardSensorFrame(
        frame_id="FRAME_BLANK_04",
        cloud_cover_pct=5.0,
        signal_to_noise_ratio=1.1,
        debris_streak_detected=False,
        raw_size_mb=25.0
    )

    decision = triage_engine.classify_frame(frame)
    assert decision == TriageDecision.DISCARD_OBSCURED_CLOUD.value
    assert triage_engine.total_downlinked_mb == 0.0


def test_savings_report_calculation(triage_engine):
    """
    4. The get_savings_report() must accurately calculate total raw MB, downlinked MB,
       and bandwidth saved percentage across a diverse stream of sensor captures.
    """
    frames = [
        # Frame 1: Cloudy -> Discard (0 MB downlinked)
        OnboardSensorFrame("F1", cloud_cover_pct=85.0, signal_to_noise_ratio=12.0, debris_streak_detected=False, raw_size_mb=50.0),
        # Frame 2: Cloudy -> Discard (0 MB downlinked)
        OnboardSensorFrame("F2", cloud_cover_pct=70.0, signal_to_noise_ratio=15.0, debris_streak_detected=False, raw_size_mb=50.0),
        # Frame 3: Blank SNR -> Discard (0 MB downlinked)
        OnboardSensorFrame("F3", cloud_cover_pct=10.0, signal_to_noise_ratio=0.8, debris_streak_detected=False, raw_size_mb=50.0),
        # Frame 4: Debris hazard -> Priority Downlink (50 MB downlinked)
        OnboardSensorFrame("F4", cloud_cover_pct=20.0, signal_to_noise_ratio=25.0, debris_streak_detected=True, raw_size_mb=50.0),
        # Frame 5: Clear science -> Standard Downlink (50 MB downlinked)
        OnboardSensorFrame("F5", cloud_cover_pct=30.0, signal_to_noise_ratio=18.0, debris_streak_detected=False, raw_size_mb=50.0),
    ]

    for f in frames:
        triage_engine.classify_frame(f)

    report = triage_engine.get_savings_report()

    # Total frames = 5
    assert report["total_frames_processed"] == 5

    # Total raw = 5 * 50 = 250 MB
    assert report["total_raw_data_mb"] == 250.0

    # Downlinked = 2 * 50 = 100 MB
    assert report["actual_downlinked_data_mb"] == 100.0

    # Saved = 250 - 100 = 150 MB
    assert report["bandwidth_saved_mb"] == 150.0

    # Saved pct = (150 / 250) * 100 = 60.0%
    assert report["bandwidth_saved_pct"] == "60.0%"

    # Check decision counts breakdown
    assert report["decision_counts"]["DISCARD_OBSCURED_CLOUD"] == 3
    assert report["decision_counts"]["PRIORITY_CONJUNCTION_HAZARD"] == 1
    assert report["decision_counts"]["STANDARD_DOWNLINK"] == 1


def test_frame_dataclass_validation():
    """Validates boundary constraints on OnboardSensorFrame."""
    with pytest.raises(ValueError, match="cloud_cover_pct must be between 0.0 and 100.0"):
        OnboardSensorFrame("ERR1", cloud_cover_pct=110.0, signal_to_noise_ratio=10.0, debris_streak_detected=False, raw_size_mb=10.0)

    with pytest.raises(ValueError, match="raw_size_mb must be strictly positive"):
        OnboardSensorFrame("ERR2", cloud_cover_pct=10.0, signal_to_noise_ratio=10.0, debris_streak_detected=False, raw_size_mb=-5.0)
