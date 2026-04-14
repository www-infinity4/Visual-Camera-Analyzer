"""Samarium Digital Twin — Sm (Z=62)
Sense-Model-Transmit loop for Samarium REE emissions.
Hardware proxy: Heat radiation
Primary use:    Neutron absorber, Sm-Co magnets
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Sm",
    atomic_number=62,
    name="Samarium",
    category=REECategory.LREE,
    primary_spectral_line_nm=359.3,
    rf_signature_type=RFSignatureType.HARMONIC,
    emission_focus="Heat radiation",
    primary_use="Neutron absorber, Sm-Co magnets",
    ore_minerals=['monazite', 'bastnäsite'],
    secondary_spectral_lines_nm=[359.3, 442.4],
    ionization_energy_ev=5.644,
    abundance_ppm_earth_crust=7.9,
    hazard_notes="",
    rf_carrier_hz=359.3,
    rf_harmonic_count=3,
)


class SamariumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
