"""Ytterbium Digital Twin — Yb (Z=70)
Sense-Model-Transmit loop for Ytterbium REE emissions.
Hardware proxy: Laser output
Primary use:    Yb-doped fibre lasers (976/1030 nm)
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Yb",
    atomic_number=70,
    name="Ytterbium",
    category=REECategory.HREE,
    primary_spectral_line_nm=398.8,
    rf_signature_type=RFSignatureType.STEADY_STATE,
    emission_focus="Laser output",
    primary_use="Yb-doped fibre lasers (976/1030 nm)",
    ore_minerals=['xenotime', 'monazite'],
    secondary_spectral_lines_nm=[328.9, 346.4],
    ionization_energy_ev=6.254,
    abundance_ppm_earth_crust=3.2,
    hazard_notes="",
    rf_carrier_hz=398.8,
    rf_harmonic_count=3,
)


class YtterbiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
