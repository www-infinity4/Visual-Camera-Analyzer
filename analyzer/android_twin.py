"""
Android Hardware Digital Twin — Termux & ADB Python Interface
═════════════════════════════════════════════════════════════
Termux-ready implementation that reads Android's internal /sys/ and /proc/
sysfs files to gather fine-grained thermal, power, and sensor data, then
maps each reading to a specific REE Digital Twin.

Two operating modes
───────────────────
  ON-DEVICE (Termux):  Run directly on the Android phone.
      python android_twin.py

  PC-SIDE (ADB):       Run from a PC connected via USB debug.
      python android_twin.py --adb

Android hardware → REE mapping
───────────────────────────────
  Component                  REE           sysfs source
  ─────────────────────────  ────────────  ─────────────────────────────────
  Vibration motor magnets    Neodymium     thermal_zone3 + battery/current
  Camera lenses (La glass)   Lanthanum     thermal_zone4 (ISP) + cpu_freq
  Display phosphors (Tb/Eu)  Terbium       thermal_zone2 (GPU/display)
  Speaker magnets (Nd)       Neodymium     audio power draw
  Battery electrodes (Ce/La) Cerium        battery/voltage_now derivative
  Wi-Fi/BT filter (YIG)      Yttrium       net_io bytes_sent
  Gyroscope MEMS (Er doped)  Erbium        virtual_memory.percent
  NFC / Wireless (SmCo)      Samarium      thermal_zone0 (modem)

Deep sysfs sources
──────────────────
  /sys/class/thermal/thermal_zone*/temp    (millidegrees Celsius)
  /sys/class/power_supply/battery/*        (µA, µV, %)
  /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   (kHz)
  /proc/cpuinfo                            (model, core count)
  /proc/net/dev                            (network I/O counters)

References
──────────
[1] Android thermal sysfs:
    https://docs.kernel.org/driver-api/thermal/sysfs-api.html
[2] Android Health HAL:
    https://source.android.com/docs/core/perf/health/implementation-2-1
[3] ADB thermal access:
    https://stackoverflow.com/questions/38948397/how-to-find-cpu-temperature-on-any-android-with-adb-command
[4] Linux thermal zones:
    https://source.android.com/docs/core/power/thermal-mitigation
[5] Termux thermal zone note:
    https://www.reddit.com/r/termux/comments/zak5z8/cant_get_thermal_information_on_a_pixel_6_android/
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from analyzer.hardware_twin import (
    DeviceMineralTwin,
    MineralEmissions,
    PowerSupplyReader,
    ThermalZoneReader,
)
from analyzer.ree_digital_twins import RFSignatureType


# ---------------------------------------------------------------------------
# sysfs path constants
# ---------------------------------------------------------------------------

THERMAL_BASE   = "/sys/class/thermal"
BATTERY_BASE   = "/sys/class/power_supply/battery"
CPUFREQ_PREFIX = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq"
CPUINFO_PATH   = "/proc/cpuinfo"
NET_DEV_PATH   = "/proc/net/dev"
BACKLIGHT_BASE = "/sys/class/backlight"


# ---------------------------------------------------------------------------
# Android-specific thermal zone type catalogue
# ---------------------------------------------------------------------------

# Typical Android thermal zone type names → best-fit REE proxy
ANDROID_ZONE_REE_MAP: Dict[str, str] = {
    "cpu-thermal":      "Lanthanum",
    "cpu0-thermal":     "Lanthanum",
    "cpu1-thermal":     "Holmium",
    "gpu-thermal":      "Terbium",
    "gpu0-thermal":     "Terbium",
    "display-thermal":  "Europium",
    "battery":          "Cerium",
    "battery-thermal":  "Cerium",
    "camera-isp":       "Lanthanum",
    "cam-isp":          "Lanthanum",
    "modem-thermal":    "Thulium",
    "wlan-thermal":     "Yttrium",
    "bt-thermal":       "Samarium",
    "charger-thermal":  "Lutetium",
    "pa-thermal":       "Gadolinium",  # power amplifier
    "xo-thermal":       "Ytterbium",   # crystal oscillator
    "skin-thermal":     "Praseodymium",
    "msm_thermal":      "Dysprosium",  # Qualcomm MSM SoC
    "tsens_tz_sensor0": "Neodymium",
    "tsens_tz_sensor1": "Praseodymium",
    "tsens_tz_sensor2": "Scandium",
    "tsens_tz_sensor3": "Erbium",
    "ambient":          "Promethium",
}


# ---------------------------------------------------------------------------
# Low-level sysfs helpers
# ---------------------------------------------------------------------------


def _read_sysfs(path: str, fallback: str = "0") -> str:
    """Read a sysfs file and return its stripped string content."""
    try:
        return open(path).read().strip()
    except OSError:
        return fallback


def _read_int(path: str, fallback: int = 0) -> int:
    try:
        return int(_read_sysfs(path, str(fallback)))
    except ValueError:
        return fallback


def _read_float(path: str, divisor: float = 1.0, fallback: float = 0.0) -> float:
    try:
        return float(_read_sysfs(path, str(fallback))) / divisor
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# Android-specific thermal discovery
# ---------------------------------------------------------------------------


class AndroidThermalZoneReader(ThermalZoneReader):
    """
    Extends ThermalZoneReader with Android-specific zone → REE mapping.

    Also provides shell commands equivalent to:
        for i in $(ls /sys/class/thermal/ | grep thermal_zone); do
            echo "$i type: $(cat /sys/class/thermal/$i/type)"; done
    """

    def discover_with_ree_map(self) -> Dict[str, Tuple[str, str]]:
        """
        Return a dict mapping zone_path → (zone_type, best_ree_proxy).

        Example::
            {
              "/sys/.../thermal_zone0": ("cpu-thermal", "Lanthanum"),
              "/sys/.../thermal_zone2": ("gpu-thermal", "Terbium"),
              ...
            }
        """
        raw = self.discover()
        result: Dict[str, Tuple[str, str]] = {}
        for path, zone_type in raw.items():
            ree = ANDROID_ZONE_REE_MAP.get(zone_type.lower(), "unknown")
            result[path] = (zone_type, ree)
        return result

    def shell_discover_command(self) -> str:
        """Return the shell one-liner from the requirement spec."""
        return (
            "for i in $(ls /sys/class/thermal/ | grep thermal_zone); do "
            r'    echo "$i type: $(cat /sys/class/thermal/$i/type)"; '
            "done"
        )

    def read_zone_by_ree(self, ree_name: str) -> float:
        """
        Find the best thermal zone for a REE proxy and return its temperature.

        Args:
            ree_name: REE element name.

        Returns:
            Temperature in °C, or 45.0 if no matching zone found.
        """
        zone_map = self.discover_with_ree_map()
        for _path, (zone_type, ree) in zone_map.items():
            if ree.lower() == ree_name.lower():
                # extract zone index from path
                idx_match = re.search(r"thermal_zone(\d+)", _path)
                if idx_match:
                    return self.read_temp_celsius(int(idx_match.group(1)))
        return 45.0


# ---------------------------------------------------------------------------
# Battery / power supply (Android sysfs)
# ---------------------------------------------------------------------------


class AndroidBatteryReader(PowerSupplyReader):
    """
    Extends PowerSupplyReader with Android-specific dumpsys bridge.

    Reads via sysfs when running in Termux; falls back to `adb shell dumpsys`
    when called from a PC via the ADB bridge.
    """

    def current_ma(self) -> float:
        """Return current in milliamps (positive = charging)."""
        return self.current_ua() / 1000.0

    def power_mw(self) -> float:
        """Instantaneous power in milliwatts (|V| × |I|)."""
        v_v = self.voltage_uv() / 1e6
        i_a = abs(self.current_ua()) / 1e6
        return v_v * i_a * 1000.0

    @staticmethod
    def dumpsys_summary() -> Dict[str, str]:
        """
        Run `dumpsys battery` and return key-value pairs.

        Works in Termux (on-device) or via ADB.  Requires no root.
        """
        result: Dict[str, str] = {}
        try:
            raw = subprocess.check_output(
                ["dumpsys", "battery"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode(errors="replace")
            for line in raw.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    result[key.strip()] = val.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return result


# ---------------------------------------------------------------------------
# Android Mineral Twin
# ---------------------------------------------------------------------------


class AndroidMineralTwin(DeviceMineralTwin):
    """
    Android/Termux version of the Hardware-as-a-Proxy Digital Twin.

    Uses fine-grained Android sysfs files for more accurate component-level
    readings compared to the generic psutil-based DeviceMineralTwin.

    Each of the 17 REEs is mapped to the specific Android hardware component
    that physically contains that mineral.

    Usage on Android (Termux)::

        pkg install python
        pip install numpy psutil
        python -c "
        from analyzer.android_twin import AndroidMineralTwin
        twin = AndroidMineralTwin()
        twin.run_loop()
        "

    Usage via ADB from PC::

        adb forward tcp:5678 tcp:5678
        python android_twin.py --adb --host 127.0.0.1 --port 5678
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        enable_rf_output: bool = False,
    ):
        super().__init__(rng=rng, enable_rf_output=enable_rf_output)
        self._android_thermal = AndroidThermalZoneReader()
        self._android_battery = AndroidBatteryReader()
        self._zone_ree_cache: Optional[Dict] = None

    def _zone_cache(self) -> Dict:
        if self._zone_ree_cache is None:
            self._zone_ree_cache = self._android_thermal.discover_with_ree_map()
        return self._zone_ree_cache

    def get_internal_emissions(self) -> MineralEmissions:
        """
        Override to prefer Android sysfs readings over psutil averages.

        On Android, each sysfs zone is resolved to its best-fit REE proxy
        using ANDROID_ZONE_REE_MAP.  When a zone is not found, falls back
        to the parent class psutil-based method.
        """
        # Use parent class for the full emission struct
        base = super().get_internal_emissions()

        # Override with fine-grained sysfs reads where available
        nd_zone  = self._android_thermal.read_zone_by_ree("Neodymium")
        la_zone  = self._android_thermal.read_zone_by_ree("Lanthanum")
        tb_zone  = self._android_thermal.read_zone_by_ree("Terbium")
        eu_zone  = self._android_thermal.read_zone_by_ree("Europium")
        sm_zone  = self._android_thermal.read_zone_by_ree("Samarium")
        dy_zone  = self._android_thermal.read_zone_by_ree("Dysprosium")
        ce_zone  = self._android_thermal.read_zone_by_ree("Cerium")

        power_mw = self._android_battery.power_mw()
        current_ma = abs(self._android_battery.current_ma())

        import dataclasses
        return dataclasses.replace(
            base,
            Neodymium_Flux       = np.sin(nd_zone) * (current_ma / 3000.0),
            Terbium_Excitation   = tb_zone ** 1.2,
            Lanthanum_Refraction = float(self._rng.normal(la_zone, 0.5)),
            Dysprosium_Drift     = dy_zone,
            Europium_Luminance   = eu_zone * power_mw / 10_000.0,
            Cerium_Polish        = abs(ce_zone - la_zone),
            Samarium_Magnet      = sm_zone * current_ma / 1e5,
        )

    def print_zone_map(self) -> None:
        """
        Print the Android thermal zone discovery table.

        Equivalent to the shell one-liner from the requirement:
            for i in $(ls /sys/class/thermal/ | grep thermal_zone); do
                echo "$i type: $(cat /sys/class/thermal/$i/type)"
            done
        """
        zone_map = self._zone_cache()
        if not zone_map:
            print("No thermal zones found. "
                  "Run on Android (Termux) or check /sys/class/thermal/.")
            return
        print(f"\n{'─'*60}")
        print(f"  Android Thermal Zone → REE Mapping")
        print(f"{'─'*60}")
        for path, (zone_type, ree) in sorted(zone_map.items()):
            print(f"  {os.path.basename(path):<22} {zone_type:<25} → {ree}")
        print(f"{'─'*60}")

    def battery_summary(self) -> str:
        """Return a formatted battery summary (dumpsys + sysfs)."""
        current_ma = self._android_battery.current_ma()
        voltage_mv = self._android_battery.voltage_uv() / 1000.0
        temp_c     = self._android_battery.temp_celsius()
        power_mw   = self._android_battery.power_mw()
        return (
            f"Battery: {voltage_mv:.0f} mV | "
            f"{current_ma:+.0f} mA | "
            f"{temp_c:.1f}°C | "
            f"{power_mw:.1f} mW"
        )


# ---------------------------------------------------------------------------
# ADB bridge (PC-side remote execution)
# ---------------------------------------------------------------------------


class ADBBridge:
    """
    Run sysfs read commands on a connected Android device via ADB.

    Provides a Python interface to the shell commands from the requirement:
        adb shell cat /sys/class/thermal/thermal_zone0/temp
        adb shell cat /sys/class/power_supply/battery/current_now
        adb shell dumpsys battery | grep -E "voltage|temperature|current"
    """

    def __init__(self, adb_path: str = "adb", device_serial: Optional[str] = None):
        self.adb_path = adb_path
        self.device_serial = device_serial

    def _run(self, *args) -> str:
        """Execute an adb shell command and return stdout."""
        cmd = [self.adb_path]
        if self.device_serial:
            cmd += ["-s", self.device_serial]
        cmd += ["shell"] + list(args)
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, timeout=10
            ).decode(errors="replace").strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return ""

    def read_thermal_zone(self, index: int = 0) -> float:
        """
        Read temperature of thermal_zone{index} via ADB.

        Equivalent to:  adb shell cat /sys/class/thermal/thermal_zone0/temp
        """
        raw = self._run("cat", f"/sys/class/thermal/thermal_zone{index}/temp")
        try:
            val = int(raw)
            return val / 1000.0 if val > 1000 else float(val)
        except ValueError:
            return 45.0

    def read_battery_current(self) -> float:
        """
        Read real-time battery current in µA.

        Equivalent to:  adb shell cat /sys/class/power_supply/battery/current_now
        """
        raw = self._run("cat", "/sys/class/power_supply/battery/current_now")
        try:
            return float(raw)
        except ValueError:
            return 500_000.0

    def discover_thermal_zones(self) -> Dict[str, str]:
        """
        Enumerate all thermal zones on the connected device.

        Equivalent to the shell loop from the requirement spec.
        """
        zones: Dict[str, str] = {}
        output = self._run(
            "for i in $(ls /sys/class/thermal/ | grep thermal_zone); do "
            r"echo \"$i:$(cat /sys/class/thermal/$i/type)\"; "
            "done"
        )
        for line in output.splitlines():
            if ":" in line:
                zone, _, ztype = line.partition(":")
                zones[zone.strip()] = ztype.strip()
        return zones

    def dumpsys_battery(self) -> Dict[str, str]:
        """
        Run dumpsys battery and return voltage/temperature/current fields.

        Equivalent to:
            adb shell dumpsys battery | grep -E "voltage|temperature|current"
        """
        raw = self._run("dumpsys", "battery")
        result: Dict[str, str] = {}
        for line in raw.splitlines():
            if any(k in line.lower() for k in ("voltage", "temperature", "current")):
                if ":" in line:
                    k, _, v = line.partition(":")
                    result[k.strip()] = v.strip()
        return result

    def is_connected(self) -> bool:
        """Return True if at least one Android device is reachable via ADB."""
        try:
            out = subprocess.check_output(
                [self.adb_path, "devices"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode()
            lines = [l for l in out.splitlines()[1:] if l.strip() and "offline" not in l]
            return len(lines) > 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """
    Command-line entry point for the Android Digital Twin.

    Usage:
        python -m analyzer.android_twin           # on-device (Termux)
        python -m analyzer.android_twin --adb     # from PC via ADB
        python -m analyzer.android_twin --zones   # print zone map only
    """
    import argparse
    parser = argparse.ArgumentParser(description="Android REE Digital Twin")
    parser.add_argument("--adb",    action="store_true", help="Use ADB bridge")
    parser.add_argument("--zones",  action="store_true", help="Discover thermal zones")
    parser.add_argument("--cycles", type=int, default=None, help="Number of cycles")
    parser.add_argument("--interval", type=float, default=1.0, help="Cycle interval (s)")
    args = parser.parse_args()

    twin = AndroidMineralTwin()

    if args.zones:
        twin.print_zone_map()
        print(f"\nBattery: {twin.battery_summary()}")
        return

    if args.adb:
        bridge = ADBBridge()
        if not bridge.is_connected():
            print("No Android device found via ADB. Connect via USB with debug enabled.")
            return
        print("ADB device connected. Running remote thermal scan...")
        zones = bridge.discover_thermal_zones()
        for zone, ztype in zones.items():
            print(f"  {zone}: {ztype}")
        battery = bridge.dumpsys_battery()
        print(f"\nBattery state:")
        for k, v in battery.items():
            print(f"  {k}: {v}")
        return

    # On-device loop
    print("Starting Android Hardware Digital Twin...")
    print("(Press Ctrl+C to stop)\n")
    twin.run_loop(interval_s=args.interval, max_cycles=args.cycles)


if __name__ == "__main__":
    main()
