"""
Onboard Edge AI Data Triage Engine.
Simulates onboard machine learning inference (FPGA / Edge TPU) on spaceborne sensors
to intelligently filter obscured or empty frames and prioritize critical hazard observations
before ground station downlink.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List


class TriageDecision(str, Enum):
    """
    Onboard Edge AI classification outcomes for sensor telemetry.
    """
    DISCARD_OBSCURED_CLOUD = "DISCARD_OBSCURED_CLOUD"
    PRIORITY_CONJUNCTION_HAZARD = "PRIORITY_CONJUNCTION_HAZARD"
    STANDARD_DOWNLINK = "STANDARD_DOWNLINK"

    def __str__(self) -> str:
        return self.value


@dataclass
class OnboardSensorFrame:
    """
    Represents an onboard raw optical or radar sensor frame captured by a satellite.

    Attributes:
        frame_id: Unique string identifier for the captured frame.
        cloud_cover_pct: Estimated cloud cover percentage (0.0 to 100.0).
        signal_to_noise_ratio: Sensor Signal-to-Noise Ratio (SNR) in dB.
        debris_streak_detected: Boolean indicating whether an optical/radar streak
                                 matching a non-cataloged space debris object is detected.
        raw_size_mb: Uncompressed file size in Megabytes (MB).
        timestamp: UTC acquisition timestamp (defaults to current UTC time).
        metadata: Optional dictionary for additional onboard sensor parameters.
    """
    frame_id: str
    cloud_cover_pct: float
    signal_to_noise_ratio: float
    debris_streak_detected: bool
    raw_size_mb: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates range constraints on sensor telemetry values."""
        if not (0.0 <= self.cloud_cover_pct <= 100.0):
            raise ValueError(
                f"cloud_cover_pct must be between 0.0 and 100.0, got {self.cloud_cover_pct}"
            )
        if self.raw_size_mb <= 0.0:
            raise ValueError(
                f"raw_size_mb must be strictly positive, got {self.raw_size_mb}"
            )


class EdgeAITriageEngine:
    """
    Simulates onboard Edge AI / FPGA processor executing real-time data triage.

    Filters uninformative data (cloud-obscured, overexposed, or noisy frames) onboard
    to prevent downlink congestion, while prioritizing time-critical conjunction observations.
    """
    def __init__(
        self,
        cloud_threshold_pct: float = 65.0,
        min_snr_threshold: float = 2.0
    ):
        """
        Initializes triage inference engine and performance accounting metrics.

        Parameters:
            cloud_threshold_pct: Cloud cover percentage above which frames are discarded (default: 65.0%).
            min_snr_threshold: Minimum acceptable SNR in dB (below which frame is blank/overexposed).
        """
        self.cloud_threshold_pct = cloud_threshold_pct
        self.min_snr_threshold = min_snr_threshold

        self.total_frames_processed: int = 0
        self.total_raw_mb: float = 0.0
        self.total_downlinked_mb: float = 0.0

        self.decision_counts: Dict[str, int] = {
            TriageDecision.DISCARD_OBSCURED_CLOUD.value: 0,
            TriageDecision.PRIORITY_CONJUNCTION_HAZARD.value: 0,
            TriageDecision.STANDARD_DOWNLINK.value: 0,
        }

    def classify_frame(self, frame: OnboardSensorFrame) -> str:
        """
        Classifies an onboard sensor capture using edge AI triage heuristics and updates telemetry stats.

        Classification Rules:
          1. DISCARD_OBSCURED_CLOUD:
             - If cloud_cover_pct > 65.0 OR signal_to_noise_ratio < 2.0 (blank / overexposed).
             - 0 MB downlinked.
          2. PRIORITY_CONJUNCTION_HAZARD:
             - If debris_streak_detected is True.
             - Full raw_size_mb downlinked immediately as high-priority telemetry.
          3. STANDARD_DOWNLINK:
             - Default for clear, high-quality science imagery.
             - Full raw_size_mb downlinked.

        Parameters:
            frame: OnboardSensorFrame telemetry capture.

        Returns:
            str: Triage decision string ('DISCARD_OBSCURED_CLOUD',
                 'PRIORITY_CONJUNCTION_HAZARD', or 'STANDARD_DOWNLINK').
        """
        self.total_frames_processed += 1
        self.total_raw_mb += frame.raw_size_mb

        # Rule 1: Priority Conjunction Hazard takes precedence if a real debris streak is detected
        # unless frame is completely blinded/obscured beyond utility
        if frame.cloud_cover_pct > self.cloud_threshold_pct or frame.signal_to_noise_ratio < self.min_snr_threshold:
            decision = TriageDecision.DISCARD_OBSCURED_CLOUD.value
            downlinked_mb = 0.0
        elif frame.debris_streak_detected:
            decision = TriageDecision.PRIORITY_CONJUNCTION_HAZARD.value
            downlinked_mb = frame.raw_size_mb
        else:
            decision = TriageDecision.STANDARD_DOWNLINK.value
            downlinked_mb = frame.raw_size_mb

        self.total_downlinked_mb += downlinked_mb
        self.decision_counts[decision] += 1

        return decision

    def get_savings_report(self) -> Dict[str, Any]:
        """
        Generates a summary report of data savings and edge triage efficiency.

        Returns:
            dict: Summary metrics containing:
                - 'total_frames_processed' (int)
                - 'total_raw_data_mb' (float)
                - 'actual_downlinked_data_mb' (float)
                - 'bandwidth_saved_mb' (float)
                - 'bandwidth_saved_pct' (str, e.g. "78.4%")
                - 'decision_counts' (dict)
        """
        saved_mb = max(0.0, self.total_raw_mb - self.total_downlinked_mb)
        pct_saved = (saved_mb / self.total_raw_mb * 100.0) if self.total_raw_mb > 0.0 else 0.0

        return {
            "total_frames_processed": self.total_frames_processed,
            "total_raw_data_mb": round(self.total_raw_mb, 2),
            "actual_downlinked_data_mb": round(self.total_downlinked_mb, 2),
            "bandwidth_saved_mb": round(saved_mb, 2),
            "bandwidth_saved_pct": f"{pct_saved:.1f}%",
            "decision_counts": dict(self.decision_counts),
        }
