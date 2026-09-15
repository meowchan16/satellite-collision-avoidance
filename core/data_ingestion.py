"""
Data Ingestion Module for Satellite and Space Debris Catalogs.
Handles ingestion of raw Two-Line Element (TLE) sets, checksum validation,
metadata classification (NORAD ID, name, object type, orbit class), and catalog querying.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


def validate_tle_checksum(line: str) -> bool:
    """
    Validates the standard NORAD TLE line checksum (modulo 10).
    The checksum is the last character of the line (index 68 for 69-char lines).
    Rules:
      - Numbers 0-9 add their face value.
      - Minus sign ('-') adds 1.
      - Plus sign ('+'), space, letters, and periods add 0.
      - Sum modulo 10 must equal the last digit.
    """
    stripped = line.strip()
    if len(stripped) < 69:
        return False
    
    # We validate the first 69 characters of the TLE line
    target_line = stripped[:69]
    try:
        expected_checksum = int(target_line[68])
    except ValueError:
        return False

    checksum_sum = 0
    for char in target_line[:68]:
        if char.isdigit():
            checksum_sum += int(char)
        elif char == '-':
            checksum_sum += 1
        # All other characters (+, letters, space, dot) count as 0

    return (checksum_sum % 10) == expected_checksum


def classify_orbit(mean_motion: float, eccentricity: float) -> str:
    """
    Classifies orbit regime based on mean motion (revolutions per day) and eccentricity:
      - LEO (Low Earth Orbit): mean_motion >= 11.25 revs/day (approx altitude < 2,000 km)
      - GEO (Geostationary / Geosynchronous): mean_motion approx 0.95 to 1.05 and ecc < 0.2
      - MEO (Medium Earth Orbit): 1.05 < mean_motion < 11.25
      - HEO (Highly Elliptical Orbit): eccentricity > 0.25
    """
    if eccentricity > 0.25:
        return "HEO"
    if 0.95 <= mean_motion <= 1.05 and eccentricity < 0.2:
        return "GEO"
    if mean_motion >= 11.25:
        return "LEO"
    return "MEO"


@dataclass
class TLEObject:
    """
    Represents a parsed satellite or debris object from a TLE set.
    """
    name: str
    norad_id: int
    line1: str
    line2: str
    object_type: str = "UNKNOWN"  # e.g., 'PAYLOAD', 'DEBRIS', 'ROCKET BODY'
    orbit_class: str = "UNKNOWN"  # e.g., 'LEO', 'MEO', 'GEO', 'HEO'
    epoch_year: Optional[int] = None
    epoch_day: Optional[float] = None
    inclination_deg: Optional[float] = None
    raan_deg: Optional[float] = None
    eccentricity: Optional[float] = None
    arg_perigee_deg: Optional[float] = None
    mean_anomaly_deg: Optional[float] = None
    mean_motion_rev_day: Optional[float] = None
    bstar: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid_checksum(self) -> bool:
        """Returns True if both line1 and line2 pass modulo 10 checksum validation."""
        return validate_tle_checksum(self.line1) and validate_tle_checksum(self.line2)


def parse_tle_lines(name: str, line1: str, line2: str, object_type: Optional[str] = None) -> TLEObject:
    """
    Parses a single 2-line or 3-line set into a structured TLEObject.
    Extracts orbital elements from standard TLE fixed columns.
    """
    line1 = line1.strip()
    line2 = line2.strip()
    name = name.strip() if name else "UNKNOWN"

    if len(line1) < 69 or len(line2) < 69:
        raise ValueError(f"TLE lines must be at least 69 characters. Line 1: {len(line1)}, Line 2: {len(line2)}")

    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("Invalid TLE format: Line 1 must start with '1 ' and Line 2 must start with '2 '")

    # Extract NORAD ID from line 1 columns 3-7 (0-indexed 2:7)
    norad_id = int(line1[2:7].strip())

    # Epoch year (col 19-20) and day (col 21-32)
    raw_year = int(line1[18:20])
    epoch_year = 1900 + raw_year if raw_year >= 57 else 2000 + raw_year
    epoch_day = float(line1[20:32])

    # BSTAR drag term (col 54-61): e.g. " 12345-4" -> 0.12345e-4
    bstar_str = line1[53:61].strip()
    try:
        if bstar_str:
            if 'e' not in bstar_str.lower():
                mantissa = bstar_str[:-2]
                exp = bstar_str[-2:]
                bstar = float(f"{mantissa}e{exp}") * 1e-5
            else:
                bstar = float(bstar_str)
        else:
            bstar = 0.0
    except Exception:
        bstar = 0.0

    # Line 2 orbital elements
    inclination = float(line2[8:16].strip())
    raan = float(line2[17:25].strip())
    # Eccentricity has implicit leading decimal point (7 digits, col 27-33)
    ecc_str = line2[26:33].strip()
    eccentricity = float("0." + ecc_str)
    arg_perigee = float(line2[34:42].strip())
    mean_anomaly = float(line2[43:51].strip())
    mean_motion = float(line2[52:63].strip())

    # Infer type from name if not provided
    inferred_type = object_type
    if not inferred_type:
        name_upper = name.upper()
        if "DEB" in name_upper or "DEBRIS" in name_upper:
            inferred_type = "DEBRIS"
        elif "R/B" in name_upper or "ROCKET BODY" in name_upper:
            inferred_type = "ROCKET BODY"
        else:
            inferred_type = "PAYLOAD"

    orbit_class = classify_orbit(mean_motion, eccentricity)

    return TLEObject(
        name=name,
        norad_id=norad_id,
        line1=line1,
        line2=line2,
        object_type=inferred_type,
        orbit_class=orbit_class,
        epoch_year=epoch_year,
        epoch_day=epoch_day,
        inclination_deg=inclination,
        raan_deg=raan,
        eccentricity=eccentricity,
        arg_perigee_deg=arg_perigee,
        mean_anomaly_deg=mean_anomaly,
        mean_motion_rev_day=mean_motion,
        bstar=bstar
    )


def parse_tle_catalog(raw_text: str, default_type: Optional[str] = None, validate_checksums: bool = True) -> List[TLEObject]:
    """
    Parses a multi-entry TLE catalog text block.
    Supports 2-line (without header name) and 3-line (name header preceding line 1 & 2) formats.
    """
    lines = [line.rstrip() for line in raw_text.strip().splitlines() if line.strip()]
    tle_objects: List[TLEObject] = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if line is line 1 of TLE
        if line.startswith("1 ") and len(line) >= 69:
            name = f"OBJECT_{line[2:7].strip()}"
            line1 = line
            if i + 1 < len(lines) and lines[i + 1].startswith("2 ") and len(lines[i + 1]) >= 69:
                line2 = lines[i + 1]
                i += 2
            else:
                i += 1
                continue
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            # 3-line format: Line 0 = Name, Line 1 = TLE 1, Line 2 = TLE 2
            name = line.strip()
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            i += 3
        else:
            i += 1
            continue

        if validate_checksums:
            if not (validate_tle_checksum(line1) and validate_tle_checksum(line2)):
                # Skip invalid checksums
                continue

        try:
            tle_obj = parse_tle_lines(name, line1, line2, object_type=default_type)
            tle_objects.append(tle_obj)
        except Exception:
            continue

    return tle_objects


class CatalogManager:
    """
    Stores and manages indexed collections of satellite and space debris objects.
    Enables querying by NORAD ID, object type, orbit regime, and conversion into simulation objects.
    """
    def __init__(self):
        self._catalog: Dict[int, TLEObject] = {}

    def add_tle(self, tle_obj: TLEObject) -> None:
        self._catalog[tle_obj.norad_id] = tle_obj

    def ingest_text(self, raw_text: str, default_type: Optional[str] = None, validate_checksums: bool = True) -> int:
        """Ingests multi-entry TLE catalog text and returns count of successfully added objects."""
        parsed_objects = parse_tle_catalog(raw_text, default_type=default_type, validate_checksums=validate_checksums)
        for obj in parsed_objects:
            self.add_tle(obj)
        return len(parsed_objects)

    def get_by_norad_id(self, norad_id: int) -> Optional[TLEObject]:
        return self._catalog.get(norad_id)

    def filter_by_type(self, object_type: str) -> List[TLEObject]:
        target = object_type.upper()
        return [obj for obj in self._catalog.values() if obj.object_type.upper() == target]

    def filter_by_orbit_class(self, orbit_class: str) -> List[TLEObject]:
        target = orbit_class.upper()
        return [obj for obj in self._catalog.values() if obj.orbit_class.upper() == target]

    def get_all(self) -> List[TLEObject]:
        return list(self._catalog.values())

    def __len__(self) -> int:
        return len(self._catalog)
