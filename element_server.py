"""
Element Server — Flask JSON Bridge API
═══════════════════════════════════════
Acts as the Digital Twin Hub:
  • Reads hardware / virtual sensor data for all 17 REEs
  • Runs the AmmoniaSuite for NH₃ / NH₄NO₃ readings
  • Runs the SensorFusionLLM for chemical identification
  • Streams live UV-camera detections via Server-Sent Events (SSE)
  • Serves the static dashboard HTML

Run from the repository root:

    python element_server.py                  # use real hardware if available
    python element_server.py --virtual        # simulated data only
    python element_server.py --port 8080      # change port (default 5000)

On Android / Termux:

    pip install flask flask-cors psutil
    python element_server.py --virtual

Then open  http://<phone-ip>:5000/  on any browser on the same Wi-Fi network.
Find your IP with:  ifconfig wlan0 | grep inet

Endpoints
─────────
GET  /                           → Serves dashboard HTML
GET  /api/elements               → JSON: all 17 REE twin readings
GET  /api/elements/<name>        → JSON: single element reading
GET  /api/ammonia?mode=gas|ion   → JSON: AmmoniaSuite readings
GET  /api/fusion?target=<chem>   → JSON: SensorFusionLLM result
GET  /api/camera/status          → JSON: UV camera detector status
GET  /stream/elements            → SSE: live element updates (1 Hz)
POST /api/scan                   → Trigger a manual camera scan cycle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

# Flask + CORS
try:
    from flask import Flask, Response, jsonify, render_template, request, stream_with_context
    from flask_cors import CORS
except ImportError:
    print(
        "Flask / flask-cors not installed.\n"
        "Install with:  pip install flask flask-cors",
        file=sys.stderr,
    )
    sys.exit(1)

# Internal modules
from analyzer.ammonia_suite import AmmoniaSuite
from analyzer.fusion import SensorBundle, SensorFusionLLM
from analyzer.hardware_twin import DeviceMineralTwin
from analyzer.rare_earth import generate_rare_earth_signal, REE_LIBRARY
from analyzer.ree_elements import (
    CeriumTwin, DysprosiumTwin, ErbiumTwin, EuropiumTwin,
    GadoliniumTwin, HolmiumTwin, LanthanumTwin, LutetiumTwin,
    NeodymiumTwin, PraseodymiumTwin, PromethiumTwin, SamariumTwin,
    ScandiumTwin, TerbiumTwin, ThuliumTwin, YtterbiumTwin, YttriumTwin,
)
from analyzer.virtual_sensors import (
    VirtualAmmoniaGasSensor,
    VirtualSpectralSensor,
    synthesize_data_for_gemma,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
CORS(app)  # allow the React dev server to call the API

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_USE_VIRTUAL: bool = True          # set via --virtual flag or auto-detected
_last_scan_result: Optional[Dict[str, Any]] = None
_scan_count: int = 0

# Instantiate all 17 REE Digital Twins
_REE_TWINS = {
    "lanthanum":     LanthanumTwin(),
    "cerium":        CeriumTwin(),
    "praseodymium":  PraseodymiumTwin(),
    "neodymium":     NeodymiumTwin(),
    "promethium":    PromethiumTwin(),
    "samarium":      SamariumTwin(),
    "europium":      EuropiumTwin(),
    "gadolinium":    GadoliniumTwin(),
    "terbium":       TerbiumTwin(),
    "dysprosium":    DysprosiumTwin(),
    "holmium":       HolmiumTwin(),
    "erbium":        ErbiumTwin(),
    "thulium":       ThuliumTwin(),
    "ytterbium":     YtterbiumTwin(),
    "lutetium":      LutetiumTwin(),
    "scandium":      ScandiumTwin(),
    "yttrium":       YttriumTwin(),
}

_hardware_twin = DeviceMineralTwin()
_ammonia_gas   = AmmoniaSuite(mode="gas")
_ammonia_ion   = AmmoniaSuite(mode="ion")
_ree_library   = REE_LIBRARY
_fusion_engine = SensorFusionLLM(use_llm=False)   # SAM fallback by default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_hardware_emissions() -> Dict[str, Any]:
    """Read real hardware proxy data (psutil / sysfs)."""
    try:
        emissions = _hardware_twin.get_internal_emissions()
        rf_matrix = _hardware_twin.generate_rf_matrix(emissions)
        return {
            "source": "hardware",
            "emissions": {k: round(float(v), 4) for k, v in emissions.items()},
            "rf_matrix": rf_matrix.tolist(),
        }
    except Exception as exc:
        return {"source": "hardware", "error": str(exc)}


def _read_virtual_emissions() -> Dict[str, Any]:
    """Generate simulated readings for all REEs."""
    result: Dict[str, Any] = {}
    for name, twin in _REE_TWINS.items():
        state = twin.sense()
        result[name] = {
            "excitation":    round(float(state.excitation_level), 4),
            "flux":          round(float(state.flux_level), 4),
            "rf_amplitude":  round(float(state.rf_amplitude), 4),
            "rf_signature":  twin.properties.rf_signature_type.value
                             if hasattr(twin.properties.rf_signature_type, "value")
                             else str(twin.properties.rf_signature_type),
            "spectral_nm":   twin.properties.primary_spectral_line_nm,
            "emission_focus": twin.properties.emission_focus,
        }
    return result


def _ree_payload(name: str) -> Dict[str, Any]:
    twin = _REE_TWINS.get(name)
    if twin is None:
        return {"error": f"Unknown element: {name}"}
    state = twin.sense()
    props = twin.properties
    # Simulated thermal/flux as milli-°C / µA for the JS mapper
    thermal_mC = int(30_000 + state.excitation_level * 40_000)
    flux_uA    = int(state.flux_level * 1_000_000)
    return {
        "name":            props.name,
        "symbol":          props.symbol,
        "atomic_number":   props.atomic_number,
        "excitation":      thermal_mC,
        "flux":            flux_uA,
        "rf_amplitude":    round(float(state.rf_amplitude), 4),
        "rf_signature":    props.rf_signature_type.value
                           if hasattr(props.rf_signature_type, "value")
                           else str(props.rf_signature_type),
        "spectral_nm":     props.primary_spectral_line_nm,
        "emission_focus":  props.emission_focus,
        "primary_use":     props.primary_use,
        "ore_minerals":    props.ore_minerals,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main dashboard HTML page."""
    return render_template("dashboard.html")


@app.route("/api/elements", methods=["GET"])
def get_all_elements():
    """Return live readings for all 17 REE Digital Twins."""
    data: Dict[str, Any] = {}
    for name in _REE_TWINS:
        data[name] = _ree_payload(name)

    # Also include hardware proxy summary if available
    if not _USE_VIRTUAL:
        data["_hardware"] = _read_hardware_emissions()

    return jsonify({"timestamp": time.time(), "elements": data})


@app.route("/api/elements/<name>", methods=["GET"])
def get_element(name: str):
    """Return live reading for a single REE (e.g. /api/elements/neodymium)."""
    payload = _ree_payload(name.lower())
    if "error" in payload:
        return jsonify(payload), 404
    return jsonify(payload)


@app.route("/api/ammonia", methods=["GET"])
def get_ammonia():
    """
    Return AmmoniaSuite readings.

    Query parameters
    ────────────────
    mode : "gas" (default) | "ion"
      gas → NH₃ electrochemical + MOS + IR layers
      ion → NH₄NO₃ ion-selective electrode
    ppm  : float (default 15.0) — simulated ambient concentration
    """
    mode = request.args.get("mode", "gas")
    ppm  = float(request.args.get("ppm", 15.0))

    suite = _ammonia_gas if mode == "gas" else _ammonia_ion
    try:
        results = suite.full_reading(ppm)
    except Exception as exc:
        results = {"error": str(exc)}

    # Also build the Gemma-ready text report
    gemma_prompt = synthesize_data_for_gemma(
        "Ammonia" if mode == "gas" else "Ammonium Nitrate",
        ppm,
    )

    return jsonify({
        "mode":         mode,
        "ppm_input":    ppm,
        "readings":     results,
        "gemma_prompt": gemma_prompt,
        "timestamp":    time.time(),
    })


@app.route("/api/fusion", methods=["GET"])
def get_fusion():
    """
    Run the SensorFusionLLM on the current virtual sensor state.

    Query parameters
    ────────────────
    target : str (default "Ammonia") — chemical to filter for
    ppm    : float (default 15.0)    — simulated gas concentration
    """
    target = request.args.get("target", "Ammonia")
    ppm    = float(request.args.get("ppm", 15.0))

    # Build a SensorBundle from current virtual sensor state
    spectral = VirtualSpectralSensor()
    sig_pairs = spectral.get_signature(target)
    wavelengths = np.array([p[0] for p in sig_pairs])
    values      = np.array([p[1] for p in sig_pairs])

    bundle = SensorBundle(
        hsi_signature=values,
        hsi_wavelengths_nm=wavelengths,
        pid_ppm=ppm * 1.08,
        gas_ppm=ppm,
        raman_shift_cm=966.0 if "nitrate" in target.lower() else 968.0,
        ims_score=min(ppm / 100.0, 1.0),
    )

    result = _fusion_engine.analyze(bundle, target_chemical=target)
    return jsonify(result.to_dict())


@app.route("/api/camera/status", methods=["GET"])
def get_camera_status():
    """Return the latest UV camera detection result."""
    global _last_scan_result, _scan_count
    if _last_scan_result is None:
        return jsonify({
            "status":       "idle",
            "scan_count":   _scan_count,
            "detections":   [],
            "timestamp":    time.time(),
        })
    return jsonify(_last_scan_result)


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    """
    Trigger a simulated camera scan cycle.

    In a real deployment this would interface with the OpenCV camera pipeline
    from main.py.  In virtual mode it returns synthesised UV detection data.
    """
    global _last_scan_result, _scan_count
    _scan_count += 1

    # Simulate a UV scan result
    np.random.seed(_scan_count)
    n_detections = int(np.random.poisson(0.8))

    detections = []
    for i in range(n_detections):
        x = int(np.random.randint(50, 550))
        y = int(np.random.randint(50, 350))
        w = int(np.random.randint(30, 120))
        h = int(np.random.randint(20, 80))
        conf = round(float(np.random.uniform(0.6, 0.99)), 3)
        detections.append({
            "id":           i,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "label":        "urine" if conf > 0.7 else "other_fluorescent",
            "confidence":   conf,
            "area_px":      w * h,
        })

    _last_scan_result = {
        "status":       "scanned",
        "scan_count":   _scan_count,
        "detections":   detections,
        "detection_count": n_detections,
        "timestamp":    time.time(),
    }
    return jsonify(_last_scan_result)


@app.route("/stream/elements", methods=["GET"])
def stream_elements():
    """
    Server-Sent Events endpoint — pushes REE element data at 1 Hz.

    The React frontend connects via:
        const evtSource = new EventSource('/stream/elements');
        evtSource.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    """
    def generate():
        while True:
            payload: Dict[str, Any] = {}
            for name in _REE_TWINS:
                payload[name] = _ree_payload(name)
            data_str = json.dumps({"timestamp": time.time(), "elements": payload})
            yield f"data: {data_str}\n\n"
            time.sleep(1.0)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    """
    Background-subtraction calibration endpoint.

    The React CalibrationModal / dashboard POST a JSON body:
        { "thermal_readings": [float, …], "flux_readings": [float, …] }

    Returns the computed baseline so the frontend's spectral worker can
    seed its rolling-window calibrator for noise subtraction.

    Integration checklist item 3:
        "The app should ask the user to point at a clear area for 2 seconds
         to calibrate the rare earth sensor noise before it starts scanning."
    """
    from analyzer.noise_calibration import ScalarBaselineCalibrator

    body = request.get_json(silent=True) or {}
    thermal_readings: list = body.get("thermal_readings", [])
    flux_readings:    list = body.get("flux_readings",    [])

    if not thermal_readings:
        # Auto-generate baseline from current virtual sensor state
        thermal_readings = [
            _ree_payload(name)["excitation"] for name in list(_REE_TWINS.keys())[:8]
        ]
        flux_readings = [
            _ree_payload(name)["flux"] for name in list(_REE_TWINS.keys())[:8]
        ]

    t_cal = ScalarBaselineCalibrator(alpha=0.2)
    f_cal = ScalarBaselineCalibrator(alpha=0.2)
    for t, f in zip(thermal_readings, flux_readings):
        t_cal.update_baseline(float(t))
        f_cal.update_baseline(float(f))

    return jsonify({
        "status":              "calibrated",
        "thermal_baseline_mC": round(t_cal.baseline or 0.0, 1),
        "flux_baseline_uA":    round(f_cal.baseline or 0.0, 1),
        "samples_used":        len(thermal_readings),
        "timestamp":           time.time(),
        "message": (
            "Background baseline captured. Subtract thermal_baseline_mC and "
            "flux_baseline_uA from live readings to isolate chemical signal."
        ),
    })


@app.route("/api/ree/library", methods=["GET"])
def get_ree_library():
    """Return the full rare-earth spectral fingerprint library as JSON."""
    result: Dict[str, Any] = {}
    for name, twin in _REE_TWINS.items():
        props = twin.properties
        # Generate a short spectral signature sample (first 20 points)
        wl_range = np.linspace(300, 2500, 200)
        sig_vals = generate_rare_earth_signal(wl_range, props.name)
        sig_pairs = list(zip(wl_range[:20].tolist(), sig_vals[:20].tolist()))
        result[name] = {
            "symbol":          props.symbol,
            "atomic_number":   props.atomic_number,
            "spectral_nm":     props.primary_spectral_line_nm,
            "secondary_nm":    props.secondary_spectral_lines_nm,
            "rf_signature":    props.rf_signature_type.value
                               if hasattr(props.rf_signature_type, "value")
                               else str(props.rf_signature_type),
            "abundance_ppm":   props.abundance_ppm_earth_crust,
            "signature_sample": [(round(w, 1), round(v, 4)) for w, v in sig_pairs[:20]],
        }
    return jsonify(result)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual Camera Analyzer — Element Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--virtual", action="store_true",
        help="Use simulated sensor data only (no real hardware required)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="TCP port to listen on",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (0.0.0.0 = all interfaces)",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM backend for /api/fusion (requires Ollama running Gemma 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _USE_VIRTUAL = args.virtual

    if args.llm:
        _fusion_engine = SensorFusionLLM(use_llm=True)
        print("LLM backend enabled — ensure Ollama + Gemma 2 is running.")
    else:
        print("Using local SAM fallback for /api/fusion (no LLM required).")

    print(f"\nStarting Element Server on http://{args.host}:{args.port}")
    print(f"Mode: {'virtual (simulated)' if _USE_VIRTUAL else 'hardware (psutil/sysfs)'}")
    print(
        "\nEndpoints:\n"
        f"  http://localhost:{args.port}/                    → Dashboard\n"
        f"  http://localhost:{args.port}/api/elements        → All 17 REEs\n"
        f"  http://localhost:{args.port}/api/ammonia         → NH₃ / NH₄NO₃\n"
        f"  http://localhost:{args.port}/api/fusion          → Fusion result\n"
        f"  http://localhost:{args.port}/stream/elements     → SSE live feed\n"
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
