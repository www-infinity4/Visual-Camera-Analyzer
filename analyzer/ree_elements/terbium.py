"""Terbium Digital Twin — Tb (Z=65)
Sense-Model-Transmit loop for Terbium REE emissions.
Hardware proxy: Heat radiation
Primary use:    Green phosphors in LEDs, Terfenol-D actuators
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Tb",
    atomic_number=65,
    name="Terbium",
    category=REECategory.HREE,
    primary_spectral_line_nm=432.6,
    rf_signature_type=RFSignatureType.STEADY_STATE,
    emission_focus="Heat radiation",
    primary_use="Green phosphors in LEDs, Terfenol-D actuators",
    ore_minerals=['monazite', 'xenotime'],
    secondary_spectral_lines_nm=[350.9, 387.4],
    ionization_energy_ev=5.864,
    abundance_ppm_earth_crust=1.2,
    hazard_notes="",
    rf_carrier_hz=432.6,
    rf_harmonic_count=3,
)


class TerbiumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
