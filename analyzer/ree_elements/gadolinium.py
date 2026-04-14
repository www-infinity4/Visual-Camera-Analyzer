"""Gadolinium Digital Twin — Gd (Z=64)
Sense-Model-Transmit loop for Gadolinium REE emissions.
Hardware proxy: Processing runoff
Primary use:    MRI contrast agents, neutron capture
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Gd",
    atomic_number=64,
    name="Gadolinium",
    category=REECategory.LREE,
    primary_spectral_line_nm=342.2,
    rf_signature_type=RFSignatureType.PULSED,
    emission_focus="Processing runoff",
    primary_use="MRI contrast agents, neutron capture",
    ore_minerals=['monazite', 'bastnäsite'],
    secondary_spectral_lines_nm=[376.8, 405.8],
    ionization_energy_ev=6.15,
    abundance_ppm_earth_crust=6.2,
    hazard_notes="",
    rf_carrier_hz=342.2,
    rf_harmonic_count=3,
)


class GadoliniumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
