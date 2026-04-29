#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from make_dashboard import build_dashboard


STATE_NAMES = [
    "v1",
    "v2",
    "v3",
    "v4",
    "e",
    "lambda",
    "y1",
    "y2",
    "y3",
    "y4",
    "ip",
    "im",
    "iL1",
    "iL2",
    "iL3",
]

SIGMAS = np.array(
    [
        0.005,
        0.005,
        0.005,
        0.005,
        0.0005,
        0.0005,
        0.0005,
        0.005,
        0.005,
        0.0005,
        0.00005,
        0.00005,
        0.00005,
        0.00005,
        0.0003,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        1.0,
    ],
    dtype=float,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_EVENT_INPUTS = {
    "event01": REPO_ROOT / "code/data/ECE6323_PROJECT_CTCHANNEL_EVENT_01_MAIN.CFG",
    "event02": REPO_ROOT / "code/data/ECE6323_PROJECT_CTCHANNEL_EVENT_02_MAIN.cfg",
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "report/outputs"


@dataclass(frozen=True)
class SolverParameters:
    n_ratio: float = 400.0
    gm: float = 0.001
    L1: float = 26.526e-6
    L2: float = 348.0e-6
    L3: float = 348.0e-6
    M23: float = 287.0e-6
    gs1: float = 1.9635
    gs2: float = 0.1497
    gs3: float = 0.1497
    r1: float = 0.005
    r2: float = 0.4469
    r3: float = 0.4469
    gb: float = 10.0
    lambda0: float = 0.1876
    i0: float = 6.09109
    L0: float = 2.36


@dataclass(frozen=True)
class AnalogChannel:
    index: int
    name: str
    unit: str
    a: float
    b: float


@dataclass(frozen=True)
class ComtradeConfig:
    station_name: str
    analog_channels: list[AnalogChannel]
    sample_rates: list[tuple[float, int]]
    file_type: str
    time_multiplier: float


@dataclass(frozen=True)
class ComtradeRecord:
    sample: int
    time_s: float
    analog: list[float]


@dataclass(frozen=True)
class SolveResult:
    time_s: float
    sample: int
    vout: float
    burden_current_a: float
    chi_square: float
    dof: int
    confidence: float
    objective: float
    iterations: int
    converged: bool
    max_abs_normalized_residual: float
    state: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray


def parse_cfg(cfg_path: Path) -> ComtradeConfig:
    lines = [line.strip() for line in cfg_path.read_text().splitlines() if line.strip()]
    if len(lines) < 6:
        raise ValueError(f"{cfg_path} does not look like a COMTRADE CFG file.")

    # Pull out the basic COMTRADE info we actually need for this project.
    station_name = lines[0].split(",")[0].strip()
    count_parts = [part.strip() for part in lines[1].split(",")]
    analog_count = 0
    for token in count_parts[1:]:
        if token.upper().endswith("A"):
            analog_count = int(token[:-1])
            break
    if analog_count <= 0:
        raise ValueError(f"Could not determine analog channel count from {cfg_path}.")

    analog_lines = lines[2 : 2 + analog_count]
    analog_channels: list[AnalogChannel] = []
    for line in analog_lines:
        parts = [part.strip() for part in line.split(",")]
        analog_channels.append(
            AnalogChannel(
                index=int(parts[0]),
                name=parts[1],
                unit=parts[4],
                a=float(parts[5]),
                b=float(parts[6]),
            )
        )

    cursor = 2 + analog_count
    nominal_freq = float(lines[cursor].split(",")[0])
    _ = nominal_freq
    cursor += 1
    rate_count = int(lines[cursor].split(",")[0])
    cursor += 1

    sample_rates: list[tuple[float, int]] = []
    for _ in range(rate_count):
        rate_parts = [part.strip() for part in lines[cursor].split(",")]
        sample_rates.append((float(rate_parts[0]), int(rate_parts[1])))
        cursor += 1

    cursor += 2  # start / trigger timestamps
    file_type = lines[cursor].split(",")[0].strip().upper()
    cursor += 1
    time_multiplier = float(lines[cursor].split(",")[0]) if cursor < len(lines) else 1.0

    return ComtradeConfig(
        station_name=station_name,
        analog_channels=analog_channels,
        sample_rates=sample_rates,
        file_type=file_type,
        time_multiplier=time_multiplier,
    )


def parse_dat(cfg: ComtradeConfig, dat_path: Path) -> list[ComtradeRecord]:
    if cfg.file_type != "ASCII":
        raise ValueError(
            f"Only ASCII COMTRADE DAT files are supported right now. Found {cfg.file_type!r}."
        )

    records: list[ComtradeRecord] = []
    with dat_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            sample = int(row[0].strip())
            timestamp_raw = float(row[1].strip())
            analog = []
            for index, channel in enumerate(cfg.analog_channels):
                raw = float(row[2 + index].strip())
                analog.append(channel.a * raw + channel.b)
            # COMTRADE timestamps here are in microseconds, so convert to seconds.
            time_s = (timestamp_raw * cfg.time_multiplier) / 1_000_000.0
            records.append(ComtradeRecord(sample=sample, time_s=time_s, analog=analog))
    if not records:
        raise ValueError(f"No samples were read from {dat_path}.")
    return records


def regularized_gamma_q(a: float, x: float) -> float:
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 1.0

    gln = math.lgamma(a)
    eps = 3.0e-14
    fpmin = 1.0e-300

    if x < a + 1.0:
        ap = a
        delta = 1.0 / a
        series = delta
        while True:
            ap += 1.0
            delta *= x / ap
            series += delta
            if abs(delta) < abs(series) * eps:
                break
        p = series * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    i = 1
    while True:
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
        i += 1
        if i > 500:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def chi_square_survival(chi_square: float, dof: int) -> float:
    if dof <= 0:
        return float("nan")
    return regularized_gamma_q(0.5 * dof, 0.5 * chi_square)


def measurement_vector(vout: float, params: SolverParameters) -> np.ndarray:
    # Only one real measured quantity is available, so the rest of z is built from it.
    ib = -params.gb * vout
    z = np.zeros(21, dtype=float)
    z[0] = vout
    z[15:20] = ib
    return z


def initial_state(vout: float, params: SolverParameters, prev_state: np.ndarray | None) -> np.ndarray:
    # Start close to the burden measurement so Newton has a reasonable first guess.
    x = np.array(prev_state, copy=True) if prev_state is not None else np.zeros(15, dtype=float)
    burden_current = -params.gb * vout
    rough_ip = -params.n_ratio * burden_current
    x[0] = vout
    x[1] = 0.0
    x[2] = vout
    x[3] = 0.0
    x[10] = rough_ip
    x[11] = 0.0 if prev_state is None else x[11]
    x[12] = burden_current
    x[13] = burden_current
    x[14] = -burden_current
    return x


def measurement_function(
    x: np.ndarray, prev_x: np.ndarray, dt: float, params: SolverParameters
) -> np.ndarray:
    # This is the set of algebraic + dynamic equations written in measurement form h(x).
    v1, v2, v3, v4, e, lam, y1, y2, y3, y4, ip, im, iL1, iL2, iL3 = x
    _, _, _, _, _, prev_lam, _, _, _, _, _, _, prev_iL1, prev_iL2, prev_iL3 = prev_x

    dlam = (lam - prev_lam) / dt
    diL1 = (iL1 - prev_iL1) / dt
    diL2 = (iL2 - prev_iL2) / dt
    diL3 = (iL3 - prev_iL3) / dt

    gme = params.gm * e
    branch_23 = params.L2 * diL2 - params.M23 * diL3
    branch_32 = params.L3 * diL3 - params.M23 * diL2
    flux_ratio = lam / params.lambda0

    h = np.zeros(21, dtype=float)
    h[0] = v3 - v4
    h[1] = -gme - im + (ip / params.n_ratio) + iL1 + params.gs1 * params.L1 * diL1
    h[2] = gme + im - (ip / params.n_ratio) - iL2 - params.gs2 * branch_23
    h[3] = -gme - im + (ip / params.n_ratio) + iL3 + params.gs3 * branch_32
    h[4] = -v1 + v2 + e + params.L1 * diL1 + params.r1 * (gme + im - (ip / params.n_ratio))
    h[5] = -v3 + v1 + params.r2 * (iL2 + params.gs2 * branch_23) + branch_23
    h[6] = -v2 + v4 + params.r3 * (iL3 + params.gs3 * branch_32) + branch_32
    h[7] = iL2 + params.gs2 * branch_23 + params.gb * (v3 - v4)
    h[8] = -iL3 - params.gs3 * branch_32 + params.gb * (v4 - v3)
    h[9] = e - dlam
    h[10] = y1 - flux_ratio**2
    h[11] = y2 - y1**2
    h[12] = y3 - y2**2
    h[13] = y4 - y3 * y1
    h[14] = im - params.i0 * flux_ratio * y4 - (lam / params.L0)
    h[15] = -params.gb * (v3 - v4)
    h[16] = iL1 + params.gs1 * params.L1 * diL1
    h[17] = iL2 + params.gs2 * branch_23
    # This derived current must match the sign convention implied by node-4 KCL.
    h[18] = iL3 + params.gs3 * branch_32
    h[19] = gme + im - (ip / params.n_ratio)
    h[20] = v4
    return h


def objective(z: np.ndarray, h: np.ndarray) -> float:
    normalized = (z - h) / SIGMAS
    return float(normalized @ normalized)


def numerical_jacobian(
    x: np.ndarray, prev_x: np.ndarray, dt: float, params: SolverParameters
) -> np.ndarray:
    # Finite differences were good enough here and easier to debug than a full symbolic Jacobian.
    base = measurement_function(x, prev_x, dt, params)
    jac = np.zeros((base.size, x.size), dtype=float)
    for i in range(x.size):
        step = 1.0e-6 * max(1.0, abs(x[i]))
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        fp = measurement_function(xp, prev_x, dt, params)
        fm = measurement_function(xm, prev_x, dt, params)
        jac[:, i] = (fp - fm) / (2.0 * step)
    return jac


def gauss_newton_step(
    z: np.ndarray,
    x0: np.ndarray,
    prev_x: np.ndarray,
    dt: float,
    params: SolverParameters,
    max_iter: int,
) -> tuple[np.ndarray, int, bool]:
    x = x0.copy()
    w_diag = 1.0 / (SIGMAS**2)
    damping = 1.0e-9
    current_obj = objective(z, measurement_function(x, prev_x, dt, params))

    for iteration in range(1, max_iter + 1):
        h = measurement_function(x, prev_x, dt, params)
        residual = z - h
        jac = numerical_jacobian(x, prev_x, dt, params)
        if (not np.all(np.isfinite(h))) or (not np.all(np.isfinite(residual))) or (not np.all(np.isfinite(jac))):
            return x, iteration, False
        if float(np.max(np.abs(jac))) > 1.0e12:
            return x, iteration, False
        jw = jac * w_diag[:, None]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            A = jac.T @ jw + damping * np.eye(x.size)
            b = jac.T @ (w_diag * residual)
        if (not np.all(np.isfinite(A))) or (not np.all(np.isfinite(b))):
            return x, iteration, False

        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If the matrix gets ugly, least squares still usually gives a usable direction.
            dx = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.all(np.isfinite(dx)):
            return x, iteration, False

        if np.linalg.norm(dx, ord=np.inf) < 1.0e-9:
            return x, iteration, True

        # Just back off the step if it made things worse.
        step_scale = 1.0
        accepted = False
        for _ in range(12):
            candidate = x + step_scale * dx
            candidate_h = measurement_function(candidate, prev_x, dt, params)
            if not np.all(np.isfinite(candidate_h)):
                step_scale *= 0.5
                continue
            candidate_obj = objective(z, candidate_h)
            if not math.isfinite(candidate_obj):
                step_scale *= 0.5
                continue
            if candidate_obj <= current_obj:
                x = candidate
                current_obj = candidate_obj
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            return x, iteration, False
        if np.linalg.norm(step_scale * dx, ord=np.inf) < 1.0e-7:
            return x, iteration, True
    return x, max_iter, False


def solve_records(
    records: Iterable[ComtradeRecord],
    analog_index: int,
    params: SolverParameters,
    max_iter: int,
) -> list[SolveResult]:
    results: list[SolveResult] = []
    previous_state = np.zeros(15, dtype=float)
    previous_time = None

    for record in records:
        vout = float(record.analog[analog_index])
        if previous_time is None:
            dt = 1.0e-6
        else:
            dt = max(record.time_s - previous_time, 1.0e-6)
        # Carry the last solution forward since adjacent samples should be close.
        guess = initial_state(vout, params, previous_state if results else None)
        estimate, iterations, converged = gauss_newton_step(
            z=measurement_vector(vout, params),
            x0=guess,
            prev_x=previous_state,
            dt=dt,
            params=params,
            max_iter=max_iter,
        )
        predicted = measurement_function(estimate, previous_state, dt, params)
        residuals = measurement_vector(vout, params) - predicted
        normalized = residuals / SIGMAS
        chi_square = float(normalized @ normalized)
        dof = len(SIGMAS) - estimate.size

        results.append(
            SolveResult(
                time_s=record.time_s,
                sample=record.sample,
                vout=vout,
                burden_current_a=-params.gb * vout,
                chi_square=chi_square,
                dof=dof,
                confidence=chi_square_survival(chi_square, dof),
                objective=objective(measurement_vector(vout, params), predicted),
                iterations=iterations,
                converged=converged,
                max_abs_normalized_residual=float(np.max(np.abs(normalized))),
                state=estimate,
                predicted=predicted,
                residuals=residuals,
            )
        )
        previous_state = estimate
        previous_time = record.time_s
    return results


def synthetic_records(count: int = 120, sample_rate_hz: float = 4800.0) -> list[ComtradeRecord]:
    records: list[ComtradeRecord] = []
    for idx in range(count):
        t = idx / sample_rate_hz
        vout = 0.18 * math.sin(2.0 * math.pi * 60.0 * t) + 0.02 * math.sin(2.0 * math.pi * 180.0 * t)
        # Add a little disturbance so the smoke test is not just a clean sinusoid.
        if 0.018 <= t <= 0.028:
            vout += 0.12 * math.sin(2.0 * math.pi * 60.0 * t)
        records.append(ComtradeRecord(sample=idx + 1, time_s=t, analog=[vout]))
    return records


def series_to_svg(xs: list[float], ys: list[float], title: str, color: str) -> str:
    width = 900
    height = 220
    pad = 36
    if not xs:
        return f"<svg width='{width}' height='{height}'></svg>"

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(max_x, min_x):
        max_x = min_x + 1.0
    if math.isclose(max_y, min_y):
        max_y = min_y + 1.0

    points = []
    for x, y in zip(xs, ys):
        px = pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)
        py = height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)
        points.append(f"{px:.1f},{py:.1f}")
    polyline = " ".join(points)

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#d0d7de"/>
      <text x="{pad}" y="24" font-size="16" font-family="Helvetica, Arial, sans-serif">{title}</text>
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#98a2b3"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#98a2b3"/>
      <polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}" />
      <text x="{pad}" y="{height-10}" font-size="12" fill="#344054">t_min={min_x:.6f}s</text>
      <text x="{width-170}" y="{height-10}" font-size="12" fill="#344054">t_max={max_x:.6f}s</text>
      <text x="{pad}" y="{pad-8}" font-size="12" fill="#344054">max={max_y:.6f}</text>
      <text x="{pad}" y="{height-pad+18}" font-size="12" fill="#344054">min={min_y:.6f}</text>
    </svg>
    """


def write_outputs(results: list[SolveResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    residuals_csv = output_dir / "residuals.csv"
    summary_json = output_dir / "summary.json"
    report_html = output_dir / "report.html"

    with results_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "time_s",
                "vout_v",
                "burden_current_a",
                "estimated_primary_current_a",
                "estimated_flux_wb",
                "estimated_ct_secondary_voltage_v",
                "estimated_magnetizing_current_a",
                "chi_square",
                "degrees_of_freedom",
                "confidence",
                "max_abs_normalized_residual",
                "iterations",
                "converged",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.sample,
                    result.time_s,
                    result.vout,
                    result.burden_current_a,
                    result.state[10],
                    result.state[5],
                    result.state[4],
                    result.state[11],
                    result.chi_square,
                    result.dof,
                    result.confidence,
                    result.max_abs_normalized_residual,
                    result.iterations,
                    int(result.converged),
                ]
            )

    with residuals_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "time_s"] + [f"measurement_{i+1}" for i in range(21)])
        for result in results:
            writer.writerow([result.sample, result.time_s, *result.residuals.tolist()])

    # Throw a few quick summary stats in json so the dashboard/report can grab them.
    summary = {
        "samples": len(results),
        "converged_samples": sum(1 for result in results if result.converged),
        "mean_confidence": float(np.mean([result.confidence for result in results])),
        "min_confidence": float(np.min([result.confidence for result in results])),
        "max_primary_current_a": float(np.max([result.state[10] for result in results])),
        "min_primary_current_a": float(np.min([result.state[10] for result in results])),
        "max_flux_wb": float(np.max([result.state[5] for result in results])),
        "max_abs_normalized_residual": float(
            np.max([result.max_abs_normalized_residual for result in results])
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    xs = [result.time_s for result in results]
    primary = [result.state[10] for result in results]
    flux = [result.state[5] for result in results]
    confidence = [result.confidence for result in results]

    report_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ECE6323 Project 2 Report</title>
  <style>
    body {{
      font-family: Helvetica, Arial, sans-serif;
      margin: 24px;
      color: #101828;
      background: #f8fafc;
    }}
    h1, h2 {{ margin-bottom: 8px; }}
    .card {{
      background: white;
      border: 1px solid #d0d7de;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid #eaecf0;
    }}
    .mono {{ font-family: ui-monospace, Menlo, monospace; }}
  </style>
</head>
<body>
  <h1>Current Instrumentation Channel DSE Report</h1>
  <div class="card">
    <h2>Summary</h2>
    <table>
      <tr><th>Samples</th><td class="mono">{summary["samples"]}</td></tr>
      <tr><th>Converged Samples</th><td class="mono">{summary["converged_samples"]}</td></tr>
      <tr><th>Mean Confidence</th><td class="mono">{summary["mean_confidence"]:.6f}</td></tr>
      <tr><th>Min Confidence</th><td class="mono">{summary["min_confidence"]:.6f}</td></tr>
      <tr><th>Max Primary Current (A)</th><td class="mono">{summary["max_primary_current_a"]:.6f}</td></tr>
      <tr><th>Min Primary Current (A)</th><td class="mono">{summary["min_primary_current_a"]:.6f}</td></tr>
      <tr><th>Max Flux (Wb)</th><td class="mono">{summary["max_flux_wb"]:.6f}</td></tr>
      <tr><th>Max |Normalized Residual|</th><td class="mono">{summary["max_abs_normalized_residual"]:.6f}</td></tr>
    </table>
  </div>
  <div class="card"><h2>Estimated Primary Current</h2>{series_to_svg(xs, primary, "Estimated Primary Current (A)", "#b42318")}</div>
  <div class="card"><h2>Estimated Core Flux</h2>{series_to_svg(xs, flux, "Estimated CT Flux (Wb)", "#175cd3")}</div>
  <div class="card"><h2>Confidence Level</h2>{series_to_svg(xs, confidence, "Chi-Square Confidence", "#067647")}</div>
  <div class="card">
    <h2>Files</h2>
    <p class="mono">results.csv<br>residuals.csv<br>summary.json</p>
  </div>
</body>
</html>
"""
    )


def resolve_dat_path(cfg_path: Path, dat_path: Path | None) -> Path:
    if dat_path is not None:
        return dat_path
    sibling = cfg_path.with_suffix(".dat")
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"Could not locate DAT file for {cfg_path}.")


def solve_comtrade_event(
    cfg_path: Path,
    dat_path: Path | None,
    channel: int,
    limit: int | None,
    max_iter: int,
    output_dir: Path,
) -> Path:
    params = SolverParameters()
    cfg_file = cfg_path.expanduser().resolve()
    dat_file = resolve_dat_path(
        cfg_file,
        dat_path.expanduser().resolve() if dat_path is not None else None,
    )
    cfg = parse_cfg(cfg_file)
    records = parse_dat(cfg, dat_file)
    if limit:
        records = records[:limit]
    chan_idx = channel - 1
    if chan_idx < 0 or chan_idx >= len(cfg.analog_channels):
        raise ValueError(f"Analog channel {channel} is out of range for {cfg_file}.")
    results = solve_records(records, chan_idx, params, max_iter)
    out_dir = output_dir.expanduser().resolve()
    write_outputs(results, out_dir)
    return out_dir


def run_single_solver(args: argparse.Namespace) -> Path:
    params = SolverParameters()
    if args.synthetic:
        records = synthetic_records(count=args.limit or 120)
        results = solve_records(records, 0, params, args.max_iter)
        output_dir = Path(args.output_dir or "outputs/synthetic").resolve()
        write_outputs(results, output_dir)
        return output_dir
    return solve_comtrade_event(
        cfg_path=Path(args.cfg),
        dat_path=Path(args.dat) if args.dat else None,
        channel=args.channel,
        limit=args.limit,
        max_iter=args.max_iter,
        output_dir=Path(args.output_dir or f"outputs/{Path(args.cfg).stem}"),
    )


def run_project_events(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    # If someone does a quick limit test, don't stomp on the real dashboard files.
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT
    if args.limit and not args.output_dir:
        output_root = DEFAULT_OUTPUT_ROOT / f"preview_limit_{args.limit}"

    event01_dir = solve_comtrade_event(
        cfg_path=DEFAULT_EVENT_INPUTS["event01"],
        dat_path=None,
        channel=args.channel,
        limit=args.limit,
        max_iter=args.max_iter,
        output_dir=output_root / "event01",
    )
    event02_dir = solve_comtrade_event(
        cfg_path=DEFAULT_EVENT_INPUTS["event02"],
        dat_path=None,
        channel=args.channel,
        limit=args.limit,
        max_iter=args.max_iter,
        output_dir=output_root / "event02",
    )
    dashboard_path = build_dashboard(
        event01_dir=event01_dir,
        event02_dir=event02_dir,
        output_path=output_root / "dashboard.html",
    )
    return event01_dir, event02_dir, dashboard_path


def open_in_browser(path: Path) -> None:
    # Basic convenience thing so running the solver pops the dashboard open.
    webbrowser.open(path.resolve().as_uri())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ECE6323 Project 2 solver for CT instrumentation-channel dynamic state estimation."
    )
    parser.add_argument("cfg", nargs="?", help="Path to COMTRADE CFG file.")
    parser.add_argument("dat", nargs="?", help="Optional path to COMTRADE DAT file.")
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="1-based analog channel index to use as vout. Defaults to channel 1.",
    )
    parser.add_argument("--limit", type=int, help="Only solve the first N samples.")
    parser.add_argument("--max-iter", type=int, default=25, help="Maximum Gauss-Newton iterations.")
    parser.add_argument("--output-dir", help="Directory for CSV/JSON/HTML outputs.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a smoke test with synthetic vout data instead of COMTRADE input.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Generate outputs without launching the dashboard in the default browser.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.synthetic and not args.cfg:
        # Default behavior for the project is to run both provided event files.
        event01_dir, event02_dir, dashboard_path = run_project_events(args)
        print(f"Event 01 report: {event01_dir / 'report.html'}")
        print(f"Event 02 report: {event02_dir / 'report.html'}")
        print(f"Dashboard written to {dashboard_path}")
        if not args.no_open_browser:
            open_in_browser(dashboard_path)
        return

    output_dir = run_single_solver(args)
    print(f"Report written to {output_dir}")
    print(f"CSV results: {output_dir / 'results.csv'}")
    print(f"HTML report: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
