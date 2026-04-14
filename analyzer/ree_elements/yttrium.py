"""Yttrium Digital Twin — Y (Z=39)
Sense-Model-Transmit loop for Yttrium REE emissions.
Hardware proxy: Phosphor emission
Primary use:    YAG lasers, Y2O3 phosphors, fuel cells
"""
from analyzer.ree_digital_twins import REECategory, REEDigitalTwin, REEProperties, RFSignatureType

PROPERTIES = REEProperties(
    symbol="Y",
    atomic_number=39,
    name="Yttrium",
    category=REECategory.HREE,
    primary_spectral_line_nm=360.1,
    rf_signature_type=RFSignatureType.HARMONIC,
    emission_focus="Phosphor emission",
    primary_use="YAG lasers, Y2O3 phosphors, fuel cells",
    ore_minerals=['xenotime', 'monazite', 'bastnäsite'],
    secondary_spectral_lines_nm=[360.1, 362.1],
    ionization_energy_ev=6.217,
    abundance_ppm_earth_crust=33.0,
    hazard_notes="",
    rf_carrier_hz=360.1,
    rf_harmonic_count=3,
)


class YttriumTwin(REEDigitalTwin):
    def __init__(self, rng=None):
        super().__init__(PROPERTIES, rng=rng)
