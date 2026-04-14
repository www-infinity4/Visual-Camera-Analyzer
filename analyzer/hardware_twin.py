"""
Hardware-as-a-Proxy Digital Twin
══════════════════════════════════
Reads the device's own internal hardware metrics and "transmutes" them into
REE mineral emission signatures, exploiting the physical fact that smartphones
contain almost all 17 rare earth elements in their components:

  Component              REE Present            Proxy Signal Used
  ─────────────────────  ─────────────────────  ─────────────────────────────
  Vibration motor        Neodymium magnets      Battery current spike
  Camera lenses/ISP      Lanthanum glass        ISP/CPU thermal zone
  Display phosphors      Terbium, Europium      GPU / backlight temp
  Speaker magnets        Neodymium, Dysprosium  CPU load × battery drain
  Mic transducer         Praseodymium, Nd       Audio subsystem power
  Wi-Fi / Antenna        Yttrium (YIG filters)  Network I/O rate
  Gyroscope MEMS         Erbium doped crystal   CPU frequency
  Battery electrodes     Cerium, La (polishing) Battery voltage curve
  Processor (doping)     Ytterbium, Holmium     Core temperature
  NFC coil               Samarium Cobalt        NFC/BT power state

Architecture
────────────
  psutil hardware read
      │
      ├─ _read_cpu_temp()         → Lanthanum / Holmium / Ytterbium proxy
      ├─ _read_battery()          → Neodymium / Cerium / Samarium proxy
      ├─ _read_cpu_freq()         → Erbium / Praseodymium proxy
      ├─ _read_net_io()           → Yttrium / Gadolinium proxy
      ├─ _read_cpu_percent()      → Dysprosium / Terbium proxy
      └─ _read_linux_thermal()    → fine-grained per-zone (Linux / Android)
          │
          └─ get_internal_emissions()   → Dict[REE_name → float]
                  │
                  └─ generate_rf_matrix()  → 3×3 numpy matrix
                          │
                          └─ run_loop()   → continuous emission broadcast

References
──────────
psutil docs:         https://psutil.readthedocs.io/
Linux thermal sysfs: https://docs.kernel.org/driver-api/thermal/sysfs-api.html
Android source AHal: https://source.android.com/docs/core/perf/health/implementation-2-1
Android thermal:     https://source.android.com/docs/core/power/thermal-mitigation
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from analyzer.ree_digital_twins import RFSignalGenerator, RFSignatureType


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_PLATFORM = platform.system()          # "Linux", "Darwin", "Windows"
_IS_ANDROID = os.path.exists("/sys/class/thermal") and (
    os.path.exists("/system/build.prop") or
    os.path.exists("/proc/version") and
    "android" in open("/proc/version").read().lower() if os.path.exists("/proc/version") else False
)


# ---------------------------------------------------------------------------
# Thermal zone reader (Linux / Android sysfs)
# ---------------------------------------------------------------------------


class ThermalZoneReader:
    """
    Reads /sys/class/thermal/thermal_zone* files on Linux and Android.

    On Android each zone maps to a hardware component:
      thermal_zone0  : CPU cluster 0
      thermal_zone1  : CPU cluster 1 / big cores
      thermal_zone2  : GPU
      thermal_zone3  : Battery
      thermal_zone4  : Camera ISP
      thermal_zone5  : Wi-Fi / modem
      ...

    Use discover() to print all available zones before assigning proxies.
    """

    THERMAL_BASE = "/sys/class/thermal"

    def discover(self) -> Dict[str, str]:
        """
        Map all available thermal zones to their type labels.

        Returns:
            Dict mapping zone path → zone type string.
            Example: {"/sys/class/thermal/thermal_zone0": "cpu-thermal"}
        """
        zones: Dict[str, str] = {}
        if not os.path.isdir(self.THERMAL_BASE):
            return zones
        for entry in sorted(os.listdir(self.THERMAL_BASE)):
            if not entry.startswith("thermal_zone"):
                continue
            zone_path = os.path.join(self.THERMAL_BASE, entry)
            type_file = os.path.join(zone_path, "type")
            try:
                zone_type = open(type_file).read().strip()
            except OSError:
                zone_type = "unknown"
            zones[zone_path] = zone_type
        return zones

    def read_temp_celsius(self, zone_index: int = 0) -> float:
        """
        Read temperature from thermal_zone{index}/temp.

        Android / Linux store temperature in milli-degrees Celsius (÷1000).

        Args:
            zone_index: Thermal zone number.

        Returns:
            Temperature in °C, or a safe fallback (45.0) if unavailable.
        """
        path = os.path.join(
            self.THERMAL_BASE, f"thermal_zone{zone_index}", "temp"
        )
        try:
            raw = int(open(path).read().strip())
            # Kernel stores millidegrees if value > 1000, else degrees
            return raw / 1000.0 if raw > 1000 else float(raw)
        except (OSError, ValueError):
            return 45.0  # safe room-temperature fallback

    def read_all_temps(self) -> List[float]:
        """Return temperatures for all discovered thermal zones."""
        if not os.path.isdir(self.THERMAL_BASE):
            return []
        temps = []
        for entry in sorted(os.listdir(self.THERMAL_BASE)):
            if entry.startswith("thermal_zone"):
                idx = int(entry.replace("thermal_zone", ""))
                temps.append(self.read_temp_celsius(idx))
        return temps


# ---------------------------------------------------------------------------
# Battery / power supply reader
# ---------------------------------------------------------------------------


class PowerSupplyReader:
    """
    Reads /sys/class/power_supply/battery/ files for micro-amp precision.

    Files used:
      current_now   – real-time current draw (µA, negative = discharging)
      voltage_now   – battery voltage (µV)
      temp          – battery temperature (tenth-degrees Celsius, ÷10)
      capacity      – state-of-charge (%)
    """

    BATTERY_BASE = "/sys/class/power_supply/battery"
    FALLBACK_PATHS = [
        "/sys/class/power_supply/BAT0",
        "/sys/class/power_supply/BAT1",
        "/sys/class/power_supply/axp20x-battery",
    ]

    def __init__(self):
        self._base = self._find_battery_path()

    def _find_battery_path(self) -> Optional[str]:
        if os.path.isdir(self.BATTERY_BASE):
            return self.BATTERY_BASE
        for p in self.FALLBACK_PATHS:
            if os.path.isdir(p):
                return p
        return None

    def _read_file(self, filename: str, divisor: float = 1.0, fallback: float = 0.0) -> float:
        if self._base is None:
            return fallback
        path = os.path.join(self._base, filename)
        try:
            return float(open(path).read().strip()) / divisor
        except (OSError, ValueError):
            return fallback

    def current_ua(self) -> float:
        """Real-time current draw in µA (negative = discharging)."""
        return self._read_file("current_now", fallback=500_000.0)

    def voltage_uv(self) -> float:
        """Battery voltage in µV."""
        return self._read_file("voltage_now", fallback=3_800_000.0)

    def temp_celsius(self) -> float:
        """Battery temperature in °C (raw is tenths of °C)."""
        return self._read_file("temp", divisor=10.0, fallback=35.0)

    def capacity_pct(self) -> float:
        """Battery state-of-charge (0–100 %)."""
        val = self._read_file("capacity", fallback=50.0)
        # psutil fallback if sysfs unavailable
        if val == 50.0:
            bat = psutil.sensors_battery()
            if bat:
                return bat.percent
        return val


# ---------------------------------------------------------------------------
# DeviceMineralTwin — main class (matches the requirement sketch)
# ---------------------------------------------------------------------------


@dataclass
class MineralEmissions:
    """Snapshot of all 17 REE emission proxies from device hardware."""
    # From requirement: three primary proxies
    Neodymium_Flux: float        # battery current × sin(core_temp)
    Terbium_Excitation: float    # core_temp ** 1.2
    Lanthanum_Refraction: float  # N(core_temp, 0.5) random sample
    # Extended proxies for remaining 14 REEs
    Dysprosium_Drift: float      # GPU / thermal_zone2 temp
    Europium_Luminance: float    # display brightness proxy (CPU % * battery %)
    Cerium_Polish: float         # battery voltage curve derivative
    Samarium_Magnet: float       # battery drain rate × core_freq
    Praseodymium_Alloy: float    # CPU frequency ratio
    Gadolinium_MRI: float        # network I/O rate (data bus heat)
    Erbium_Fibre: float          # memory bandwidth proxy
    Ytterbium_Laser: float       # thermal zone 4 (ISP / camera)
    Holmium_Medical: float       # thermal zone 1 (big CPU cores)
    Thulium_Surgical: float      # thermal zone 5 (modem / Wi-Fi)
    Lutetium_Alloy: float        # thermal zone 3 (battery zone)
    Scandium_Aerospace: float    # CPU core count × frequency
    Yttrium_Phosphor: float      # net bytes sent (antenna proxy)
    Promethium_Decay: float      # time-decaying random (radioactive proxy)
    timestamp: float = field(default_factory=time.time)


class DeviceMineralTwin:
    """
    Hardware-as-a-Proxy Digital Twin for all 17 REE minerals.

    Maps each element to a specific device hardware component and reads
    real-time internal state via psutil and Linux /sys/ sysfs files.

    Quick start::

        twin = DeviceMineralTwin()
        while True:
            emissions = twin.get_internal_emissions()
            matrix    = twin.generate_rf_matrix(emissions)
            print(f"Mineral Emission Matrix:\\n{matrix}")
            time.sleep(1)
    """

    # Hardware-to-REE mapping (matches requirement's ree_map)
    REE_MAP: Dict[str, Dict] = {
        "Neodymium":    {"source": "battery_stats", "type": "current_draw"},
        "Terbium":      {"source": "thermal_zone",  "index": 0},
        "Lanthanum":    {"source": "cpu_freq",       "type": "voltage"},
        # Extended
        "Dysprosium":   {"source": "thermal_zone",  "index": 2},
        "Europium":     {"source": "cpu_percent",   "type": "brightness"},
        "Cerium":       {"source": "battery_stats", "type": "voltage_delta"},
        "Samarium":     {"source": "thermal_zone",  "index": 3},
        "Praseodymium": {"source": "cpu_freq",      "type": "ratio"},
        "Gadolinium":   {"source": "net_io",        "type": "bytes_recv"},
        "Erbium":       {"source": "virtual_mem",   "type": "bandwidth"},
        "Ytterbium":    {"source": "thermal_zone",  "index": 4},
        "Holmium":      {"source": "thermal_zone",  "index": 1},
        "Thulium":      {"source": "thermal_zone",  "index": 5},
        "Lutetium":     {"source": "thermal_zone",  "index": 3},
        "Scandium":     {"source": "cpu_freq",      "type": "core_count"},
        "Yttrium":      {"source": "net_io",        "type": "bytes_sent"},
        "Promethium":   {"source": "decay_model",   "type": "radioactive"},
    }

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        enable_rf_output: bool = False,
        sample_rate_hz: float = 44100.0,
    ):
        self._rng = rng or np.random.default_rng()
        self._thermal = ThermalZoneReader()
        self._power = PowerSupplyReader()
        self._rf_gen = RFSignalGenerator(sample_rate_hz=sample_rate_hz, rng=self._rng)
        self.enable_rf_output = enable_rf_output
        self._prev_voltage: float = 3.8
        self._decay_phase: float = 0.0

    # ------------------------------------------------------------------
    # Private hardware readers
    # ------------------------------------------------------------------

    def _cpu_temp(self) -> float:
        """Best-effort CPU temperature (°C)."""
        # Try psutil first (works on Linux, macOS, some Windows)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                    if key in temps and temps[key]:
                        return float(temps[key][0].current)
                # Take any first available sensor
                first_key = next(iter(temps))
                if temps[first_key]:
                    return float(temps[first_key][0].current)
        except (AttributeError, NotImplementedError):
            pass
        # Fall back to Linux sysfs
        return self._thermal.read_temp_celsius(0)

    def _battery(self) -> Tuple[float, float, float]:
        """Returns (current_ua, voltage_uv, capacity_pct)."""
        try:
            bat = psutil.sensors_battery()
            capacity = bat.percent if bat else 50.0
        except (AttributeError, NotImplementedError):
            capacity = 50.0
        current_ua = self._power.current_ua()
        voltage_uv = self._power.voltage_uv()
        return current_ua, voltage_uv, capacity

    def _cpu_freq_mhz(self) -> float:
        """Current CPU frequency in MHz."""
        try:
            freq = psutil.cpu_freq()
            return float(freq.current) if freq else 1800.0
        except (AttributeError, NotImplementedError):
            return 1800.0

    def _net_bytes(self) -> Tuple[float, float]:
        """Total network bytes (sent, recv) since boot."""
        try:
            net = psutil.net_io_counters()
            return float(net.bytes_sent), float(net.bytes_recv)
        except (AttributeError, NotImplementedError):
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Core API  (matches requirement sketch)
    # ------------------------------------------------------------------

    def get_internal_emissions(self) -> MineralEmissions:
        """
        Read the "heartbeat" of the hardware and transmute it into
        mineral emission signatures.

        Returns:
            MineralEmissions dataclass with one float per REE element.
        """
        core_temp    = self._cpu_temp()
        current_ua, voltage_uv, capacity = self._battery()
        freq_mhz     = self._cpu_freq_mhz()
        cpu_pct      = float(psutil.cpu_percent(interval=None))
        bytes_sent, bytes_recv = self._net_bytes()

        # ── Voltage delta (derivative) for Cerium ──────────────────────
        voltage_mv = voltage_uv / 1000.0
        voltage_delta = abs(voltage_mv - self._prev_voltage)
        self._prev_voltage = voltage_mv

        # ── Promethium decay model (radioactive proxy) ──────────────────
        self._decay_phase += 0.1
        promethium = float(np.exp(-self._decay_phase / 10.0) *
                           abs(np.sin(self._decay_phase)))
        if self._decay_phase > 30:
            self._decay_phase = 0.0  # reset decay cycle

        # ── Build emissions dict (from requirement) ─────────────────────
        noise = lambda: float(self._rng.normal(0, 0.5))

        emissions = MineralEmissions(
            # Requirement's three primary proxies
            Neodymium_Flux       = np.sin(core_temp) * (capacity / 100.0),
            Terbium_Excitation   = core_temp ** 1.2,
            Lanthanum_Refraction = float(self._rng.normal(core_temp, 0.5)),
            # Extended proxies
            Dysprosium_Drift     = self._thermal.read_temp_celsius(2),
            Europium_Luminance   = cpu_pct * capacity / 10_000.0,
            Cerium_Polish        = voltage_delta,
            Samarium_Magnet      = (abs(current_ua) / 1e6) * (freq_mhz / 1000.0),
            Praseodymium_Alloy   = freq_mhz / 3000.0,          # 0–1 ratio
            Gadolinium_MRI       = bytes_recv / 1e9,            # GB scale
            Erbium_Fibre         = float(psutil.virtual_memory().percent) / 100.0,
            Ytterbium_Laser      = self._thermal.read_temp_celsius(4),
            Holmium_Medical      = self._thermal.read_temp_celsius(1),
            Thulium_Surgical     = self._thermal.read_temp_celsius(5),
            Lutetium_Alloy       = self._thermal.read_temp_celsius(3),
            Scandium_Aerospace   = psutil.cpu_count(logical=False) or 4.0 * freq_mhz / 1e4,
            Yttrium_Phosphor     = bytes_sent / 1e9,
            Promethium_Decay     = promethium,
        )
        return emissions

    def generate_rf_matrix(self, emissions: MineralEmissions) -> np.ndarray:
        """
        Convert internal heat/power readings into a 3×3 RF signature matrix.

        The diagonal encodes the three primary REE proxies; off-diagonals
        encode coupling terms (cross-element interference).

        Args:
            emissions: Output of get_internal_emissions().

        Returns:
            3×3 float64 numpy matrix ready for GNU Radio / USRP amplitude
            and frequency modulation.
        """
        matrix = np.array([
            [emissions.Neodymium_Flux,       0.0,                         0.0],
            [0.0,                             emissions.Terbium_Excitation, 0.0],
            [0.0,                             0.0,                         emissions.Lanthanum_Refraction],
        ])
        # Add subtle off-diagonal coupling (cross-component EM interference)
        coupling = 0.01
        matrix[0, 1] = emissions.Samarium_Magnet   * coupling
        matrix[1, 2] = emissions.Europium_Luminance * coupling
        matrix[2, 0] = emissions.Dysprosium_Drift   * coupling
        return matrix

    def generate_rf_signal(
        self,
        emissions: MineralEmissions,
        element: str = "Neodymium",
        rf_type: RFSignatureType = RFSignatureType.STEADY_STATE,
    ) -> np.ndarray:
        """
        Generate a baseband RF waveform from a specific element's emission.

        The emission value is used as the signal amplitude — directly
        compatible with GNU Radio's signal source amplitude parameter.

        Args:
            emissions: Output of get_internal_emissions().
            element:   REE name to generate RF for.
            rf_type:   Waveform type.

        Returns:
            1-D float32 RF signal array.
        """
        value = getattr(emissions, f"{element}_Flux",
                getattr(emissions, f"{element}_Excitation",
                getattr(emissions, f"{element}_Refraction", 0.5)))
        amplitude = float(np.clip(abs(value) / 100.0, 0.0, 1.0))
        return self._rf_gen.generate(rf_type, amplitude=amplitude)

    def to_text_summary(self, emissions: MineralEmissions) -> str:
        """Structured text summary for Sensor-LLM fusion."""
        return (
            f"Hardware twin: "
            f"Nd_flux={emissions.Neodymium_Flux:.4f}, "
            f"Tb_excit={emissions.Terbium_Excitation:.2f}, "
            f"La_refr={emissions.Lanthanum_Refraction:.2f}, "
            f"Dy_drift={emissions.Dysprosium_Drift:.1f}°C, "
            f"Pm_decay={emissions.Promethium_Decay:.4f}."
        )

    def discover_thermal_zones(self) -> Dict[str, str]:
        """Print all available thermal zones (matches requirement shell cmd)."""
        return self._thermal.discover()

    # ------------------------------------------------------------------
    # Continuous loop (matches requirement's while True pattern)
    # ------------------------------------------------------------------

    def run_loop(
        self,
        interval_s: float = 1.0,
        max_cycles: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Run the Sense → Transmute → Output loop continuously.

        Args:
            interval_s:  Sleep time between cycles (seconds).
            max_cycles:  Stop after N cycles.  None = run forever.
            verbose:     Print the matrix each cycle.
        """
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            data = self.get_internal_emissions()
            matrix = self.generate_rf_matrix(data)
            if verbose:
                print(f"\n[cycle {cycle+1}] Current Mineral Emission Matrix:")
                print(np.round(matrix, 4))
            time.sleep(interval_s)
            cycle += 1
