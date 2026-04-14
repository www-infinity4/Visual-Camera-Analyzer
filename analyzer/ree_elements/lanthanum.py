"""Lanthanum Digital Twin — La (Z=57)
Sense-Model-Transmit loop for Lanthanum REE emissions.
Hardware proxy: Processing runoff
Primary use:    Camera lens high-refractive glass
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="La",
    atomic_number=57,
    name="Lanthanum",
    category=REECategory.LREE,
    primary_spectral_line_nm=394.9,
    rf_signature_type=RFSignatureType.WIDEBAND_NOISE,
    emission_focus="Processing runoff",
    primary_use="Camera lens high-refractive glass",
    ore_minerals=['monazite', 'bastnäsite', 'loparite'],
    secondary_spectral_lines_nm=[403.0, 408.7],
    ionization_energy_ev=5.577,
    abundance_ppm_earth_crust=39.0,
    hazard_notes="",
    rf_carrier_hz=394.9,
    rf_harmonic_count=3,
)


class LanthanumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
