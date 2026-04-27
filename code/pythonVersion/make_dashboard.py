#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def load_summary(summary_path: Path) -> dict[str, float]:
    return json.loads(summary_path.read_text())


def to_series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def polyline(xs: list[float], ys: list[float], x0: int, y0: int, width: int, height: int) -> tuple[str, float, float]:
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    points = []
    for x, y in zip(xs, ys):
        px = x0 + ((x - min_x) / (max_x - min_x)) * width
        py = y0 + height - ((y - min_y) / (max_y - min_y)) * height
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points), min_y, max_y


def scale_y(value: float, min_y: float, max_y: float, y0: int, height: int) -> float:
    if max_y == min_y:
        max_y = min_y + 1.0
    return y0 + height - ((value - min_y) / (max_y - min_y)) * height


def make_chart(title: str, xs: list[float], ys: list[float], color: str) -> str:
    width = 560
    height = 240
    pad = 38
    plot_x = pad
    plot_y = 28
    plot_w = width - 2 * pad
    plot_h = height - 62
    pts, min_y, max_y = polyline(xs, ys, plot_x, plot_y, plot_w, plot_h)
    tick_count = 5
    ticks = []
    for i in range(tick_count):
        ratio = i / (tick_count - 1)
        value = max_y - ratio * (max_y - min_y)
        py = scale_y(value, min_y, max_y, plot_y, plot_h)
        ticks.append((value, py))
    grid = "".join(
        f'<line x1="{plot_x}" y1="{py:.1f}" x2="{plot_x + plot_w}" y2="{py:.1f}" stroke="#e8dfcd" stroke-width="1" />'
        for _, py in ticks
    )
    labels = "".join(
        f'<text x="{plot_x - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#5f5747">{value:.4g}</text>'
        for value, py in ticks
    )
    return f"""
    <section class="chart-card">
      <div class="chart-title">{title}</div>
      <svg viewBox="0 0 {width} {height}" class="chart-svg" aria-label="{title}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#fffdf7" stroke="#d9cfb8"/>
        {grid}
        <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#a89d85"/>
        <line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#a89d85"/>
        <polyline fill="none" stroke="{color}" stroke-width="2.6" points="{pts}" />
        {labels}
        <text x="{plot_x}" y="{plot_y - 8}" font-size="11" fill="#5f5747">min={min_y:.6f} max={max_y:.6f}</text>
        <text x="{plot_x}" y="{height - 12}" font-size="11" fill="#5f5747">time (s)</text>
      </svg>
    </section>
    """


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
    </div>
    """


def build_event_panel(event_id: str, title: str, rows: list[dict[str, str]], summary: dict[str, float], active: bool) -> str:
    times = to_series(rows, "time_s")
    charts = [
        make_chart("Burden Voltage", times, to_series(rows, "vout_v"), "#b93815"),
        make_chart("Estimated Primary Current", times, to_series(rows, "estimated_primary_current_a"), "#0f766e"),
        make_chart("Estimated Core Flux", times, to_series(rows, "estimated_flux_wb"), "#1d4ed8"),
        make_chart("Confidence", times, to_series(rows, "confidence"), "#7c2d12"),
        make_chart("Max Abs Normalized Residual", times, to_series(rows, "max_abs_normalized_residual"), "#9333ea"),
    ]
    metrics = [
        metric_card("Samples", f"{int(summary['samples'])}"),
        metric_card("Converged", f"{int(summary['converged_samples'])}"),
        metric_card("Mean Confidence", f"{summary['mean_confidence']:.6f}"),
        metric_card("Max Primary Current", f"{summary['max_primary_current_a']:.3f} A"),
        metric_card("Min Primary Current", f"{summary['min_primary_current_a']:.3f} A"),
        metric_card("Max Flux", f"{summary['max_flux_wb']:.6f} Wb"),
    ]
    active_class = " active" if active else ""
    return f"""
    <section id="{event_id}" class="tab-panel{active_class}">
      <div class="hero-card">
        <div>
          <div class="eyebrow">ECE6323 Project 2</div>
          <h2>{title}</h2>
          <p>Dynamic state estimation results for the provided COMTRADE event. Use the charts below to inspect burden voltage, estimated primary current, flux, confidence, and residual behavior over time.</p>
        </div>
      </div>
      <div class="metric-grid">
        {''.join(metrics)}
      </div>
      <div class="chart-grid">
        {''.join(charts)}
      </div>
    </section>
    """


def main() -> None:
    root = Path("/Users/lpeacock/Documents/ECE6323/taketwo")
    event01_rows = load_rows(root / "outputs/event01/results.csv")
    event02_rows = load_rows(root / "outputs/event02/results.csv")
    event01_summary = load_summary(root / "outputs/event01/summary.json")
    event02_summary = load_summary(root / "outputs/event02/summary.json")

    out = root / "outputs/dashboard.html"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ECE6323 Event Dashboard</title>
  <style>
    :root {{
      --paper: #f7f2e8;
      --ink: #201a12;
      --muted: #6a6254;
      --card: #fffdf7;
      --line: #d9cfb8;
      --accent: #b93815;
      --accent-soft: #efe2d1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8ee 0, #f7f2e8 40%, #efe7d7 100%);
    }}
    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .masthead {{
      background: linear-gradient(135deg, rgba(185,56,21,0.08), rgba(32,26,18,0.02));
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 28px 32px;
      box-shadow: 0 10px 30px rgba(32, 26, 18, 0.06);
    }}
    .masthead h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}
    .masthead p {{
      margin: 0;
      max-width: 820px;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.5;
    }}
    .tab-bar {{
      display: flex;
      gap: 12px;
      margin: 24px 0 18px;
      flex-wrap: wrap;
    }}
    .tab-button {{
      appearance: none;
      border: 1px solid var(--line);
      background: rgba(255, 253, 247, 0.72);
      color: var(--ink);
      padding: 12px 18px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 0.98rem;
      transition: 180ms ease;
    }}
    .tab-button.active {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      box-shadow: 0 8px 20px rgba(185, 56, 21, 0.22);
    }}
    .tab-panel {{
      display: none;
      animation: rise 240ms ease;
    }}
    .tab-panel.active {{
      display: block;
    }}
    .hero-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px 26px;
      box-shadow: 0 10px 24px rgba(32, 26, 18, 0.05);
    }}
    .eyebrow {{
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 8px;
    }}
    .hero-card h2 {{
      margin: 0 0 10px;
      font-size: 2rem;
    }}
    .hero-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      max-width: 880px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
    }}
    .metric-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 6px 18px rgba(32, 26, 18, 0.04);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 6px;
    }}
    .metric-value {{
      font-size: 1.35rem;
      line-height: 1.1;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .chart-card {{
      background: transparent;
    }}
    .chart-title {{
      margin: 0 0 8px 6px;
      font-size: 1.02rem;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
      filter: drop-shadow(0 8px 14px rgba(32, 26, 18, 0.05));
    }}
    .footer-note {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="masthead">
      <h1>Instrumentation Channel Event Dashboard</h1>
      <p>Tabbed visual summary of the dynamic state estimation outputs for the two independent current-instrumentation events. Each tab focuses on one event so the waveforms and estimator behavior are easier to read.</p>
    </section>

    <nav class="tab-bar" aria-label="Event Tabs">
      <button class="tab-button active" data-target="event01">Event 01</button>
      <button class="tab-button" data-target="event02">Event 02</button>
    </nav>

    {build_event_panel("event01", "Event 01", event01_rows, event01_summary, True)}
    {build_event_panel("event02", "Event 02", event02_rows, event02_summary, False)}

    <p class="footer-note">Generated from <code>results.csv</code> and <code>summary.json</code> for each event.</p>
  </main>

  <script>
    const buttons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.tab-panel');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const target = button.dataset.target;
        buttons.forEach((item) => item.classList.toggle('active', item === button));
        panels.forEach((panel) => panel.classList.toggle('active', panel.id === target));
      }});
    }});
  </script>
</body>
</html>
"""
    out.write_text(html)
    print(out)


if __name__ == "__main__":
    main()
