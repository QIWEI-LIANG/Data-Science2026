from __future__ import annotations

import ast
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "visualizations"

LOCALITIES_CSV = PROCESSED / "top5_17ree_localities_analysis_ready.csv"
ELEMENTS_CSV = PROCESSED / "ree_17_elements.csv"
CLUSTER_CSV = PROCESSED / "ree_locality_clusters.csv"

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4c78a8",
    "#f58518",
]

ELEMENT_COLORS = {
    "Ce": "#f58518",
    "Gd": "#54a24b",
    "La": "#eeca3b",
    "Nd": "#b279a2",
    "Sc": "#4c78a8",
    "Sm": "#ff9da6",
    "Y": "#72b7b2",
    "Yb": "#9d755d",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_elements(value: str) -> set[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return set()
    if isinstance(parsed, list):
        return {str(item) for item in parsed}
    return set()


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_data() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    for row in read_csv(LOCALITIES_CSV):
        lat = as_float(row["latitude"])
        lon = as_float(row["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if lat == 0 and lon == 0:
            continue
        records.append(
            {
                "reserve_rank": int(row["reserve_rank"]),
                "query_country": row["query_country"],
                "reserves_reo_tonnes": int(float(row["reserves_reo_tonnes"])),
                "query_element": row["query_element"],
                "element_name": row["element_name"],
                "ree_group": row["ree_group"],
                "locality_id": row["locality_id"],
                "locality_name": row["locality_name"],
                "country": row["country"],
                "latitude": lat,
                "longitude": lon,
                "reported_elements": parse_elements(row["reported_elements"]),
                "mindat_locality_url": row["mindat_locality_url"],
            }
        )

    by_locality: dict[str, dict] = {}
    for rec in records:
        item = by_locality.setdefault(
            rec["locality_id"],
            {
                "locality_id": rec["locality_id"],
                "locality_name": rec["locality_name"],
                "country": rec["country"],
                "latitude": rec["latitude"],
                "longitude": rec["longitude"],
                "query_countries": set(),
                "query_elements": set(),
                "reported_elements": set(),
                "ree_groups": set(),
                "mindat_locality_url": rec["mindat_locality_url"],
            },
        )
        item["query_countries"].add(rec["query_country"])
        item["query_elements"].add(rec["query_element"])
        item["reported_elements"].update(rec["reported_elements"])
        item["ree_groups"].add(rec["ree_group"])

    localities = list(by_locality.values())
    return records, localities


def load_ree_symbols() -> list[str]:
    return [row["symbol"] for row in read_csv(ELEMENTS_CSV)]


def standardize(matrix: list[list[float]]) -> list[list[float]]:
    cols = len(matrix[0])
    means = [sum(row[i] for row in matrix) / len(matrix) for i in range(cols)]
    stds = []
    for i in range(cols):
        var = sum((row[i] - means[i]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(var) or 1.0)
    return [[(row[i] - means[i]) / stds[i] for i in range(cols)] for row in matrix]


def squared_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans_once(data: list[list[float]], k: int, rng: random.Random) -> tuple[list[int], list[list[float]], float]:
    centroids = [data[i][:] for i in rng.sample(range(len(data)), k)]
    labels = [0] * len(data)

    for _ in range(100):
        changed = False
        for i, row in enumerate(data):
            label = min(range(k), key=lambda c: squared_distance(row, centroids[c]))
            if label != labels[i]:
                labels[i] = label
                changed = True

        sums = [[0.0] * len(data[0]) for _ in range(k)]
        counts = [0] * k
        for label, row in zip(labels, data):
            counts[label] += 1
            for j, value in enumerate(row):
                sums[label][j] += value

        for c in range(k):
            if counts[c] == 0:
                centroids[c] = data[rng.randrange(len(data))][:]
            else:
                centroids[c] = [value / counts[c] for value in sums[c]]

        if not changed:
            break

    inertia = sum(squared_distance(row, centroids[label]) for row, label in zip(data, labels))
    return labels, centroids, inertia


def kmeans(data: list[list[float]], k: int, seed: int = 42) -> tuple[list[int], float]:
    rng = random.Random(seed)
    best_labels: list[int] | None = None
    best_inertia = float("inf")
    for _ in range(25):
        labels, _, inertia = kmeans_once(data, k, rng)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    assert best_labels is not None
    return best_labels, best_inertia


def silhouette_score(data: list[list[float]], labels: list[int]) -> float:
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    if len(clusters) < 2:
        return -1.0

    total = 0.0
    for i, row in enumerate(data):
        own = labels[i]
        own_members = clusters[own]
        if len(own_members) <= 1:
            a = 0.0
        else:
            a = sum(math.sqrt(squared_distance(row, data[j])) for j in own_members if j != i) / (len(own_members) - 1)

        b = min(
            sum(math.sqrt(squared_distance(row, data[j])) for j in members) / len(members)
            for label, members in clusters.items()
            if label != own
        )
        total += (b - a) / max(a, b) if max(a, b) else 0.0
    return total / len(data)


def cluster_localities(localities: list[dict], symbols: list[str]) -> dict:
    features = []
    for loc in localities:
        row = [loc["latitude"], loc["longitude"]]
        row.extend(1.0 if symbol in loc["reported_elements"] or symbol in loc["query_elements"] else 0.0 for symbol in symbols)
        features.append(row)

    scaled = standardize(features)
    candidates = []
    for k in range(3, 11):
        labels, inertia = kmeans(scaled, k)
        score = silhouette_score(scaled, labels)
        candidates.append({"k": k, "labels": labels, "inertia": inertia, "silhouette": score})

    best = max(candidates, key=lambda item: item["silhouette"])
    for loc, label in zip(localities, best["labels"]):
        loc["cluster"] = int(label)

    with CLUSTER_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "locality_id",
            "locality_name",
            "country",
            "latitude",
            "longitude",
            "cluster",
            "query_elements",
            "reported_ree_elements",
            "mindat_locality_url",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for loc in sorted(localities, key=lambda x: (x["cluster"], x["country"], x["locality_name"])):
            writer.writerow(
                {
                    "locality_id": loc["locality_id"],
                    "locality_name": loc["locality_name"],
                    "country": loc["country"],
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "cluster": loc["cluster"],
                    "query_elements": "; ".join(sorted(loc["query_elements"])),
                    "reported_ree_elements": "; ".join(symbol for symbol in symbols if symbol in loc["reported_elements"]),
                    "mindat_locality_url": loc["mindat_locality_url"],
                }
            )

    return {
        "best_k": best["k"],
        "best_silhouette": best["silhouette"],
        "candidates": [{k: v for k, v in c.items() if k != "labels"} for c in candidates],
    }


def page(title: str, body: str, scripts: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1e293b;
      --muted: #64748b;
      --line: #d7dde8;
      --panel: #ffffff;
      --accent: #2563eb;
      --bg: #f6f8fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    h2 {{ margin: 24px 0 12px; font-size: 18px; }}
    p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.55; }}
    main {{ padding: 20px 28px 32px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
    .metric-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #fff; min-width: 150px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 12px 0; font-size: 13px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
    svg {{ max-width: 100%; height: auto; display: block; }}
    .bar-label {{ font-size: 12px; fill: #334155; }}
    .axis-label {{ font-size: 12px; fill: #64748b; }}
    .tooltip {{
      position: fixed;
      z-index: 9999;
      max-width: 320px;
      background: #0f172a;
      color: #fff;
      padding: 8px 10px;
      border-radius: 6px;
      pointer-events: none;
      font-size: 12px;
      line-height: 1.4;
      opacity: 0;
      transition: opacity .08s ease;
    }}
    @media (max-width: 700px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      h1 {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>
{body}
{scripts}
</body>
</html>
"""


def write_index(records: list[dict], localities: list[dict], cluster_info: dict) -> None:
    body = f"""
<header>
  <h1>REE Mineral Visualization Dashboard</h1>
  <p>World distribution, clustering results, and supporting charts generated from the processed Mindat REE locality dataset.</p>
  <div class="metric-row">
    <div class="metric"><strong>{len(records)}</strong>mineral records</div>
    <div class="metric"><strong>{len(localities)}</strong>unique localities</div>
    <div class="metric"><strong>{cluster_info['best_k']}</strong>KMeans clusters</div>
    <div class="metric"><strong>{cluster_info['best_silhouette']:.3f}</strong>silhouette score</div>
  </div>
</header>
<main>
  <section class="grid">
    <div class="panel">
      <h2>1. World Distribution Map</h2>
      <p>Interactive map colored by queried mineral element. Use it to show where REE minerals appear globally.</p>
      <a href="world_mineral_distribution_map.html">Open the map</a>
    </div>
    <div class="panel">
      <h2>2. Clustering Result</h2>
      <p>Cluster map and longitude-latitude scatter colored by cluster label.</p>
      <a href="cluster_results.html">Open clustering results</a>
    </div>
    <div class="panel">
      <h2>3. Supporting Charts</h2>
      <p>Counts by cluster, mineral element, country, and REE group.</p>
      <a href="auxiliary_charts.html">Open supporting charts</a>
    </div>
  </section>
</main>
"""
    (OUT / "index.html").write_text(page("REE Visualization Dashboard", body), encoding="utf-8")


def write_world_map(records: list[dict]) -> None:
    data = [
        {
            "lat": r["latitude"],
            "lon": r["longitude"],
            "element": r["query_element"],
            "element_name": r["element_name"],
            "group": r["ree_group"],
            "country": r["query_country"],
            "site_country": r["country"],
            "name": r["locality_name"],
            "url": r["mindat_locality_url"],
        }
        for r in records
    ]
    elements = sorted({r["query_element"] for r in records})
    countries = sorted({r["query_country"] for r in records})
    legend = "".join(
        f'<span><i class="swatch" style="background:{ELEMENT_COLORS.get(el, "#555")}"></i>{html.escape(el)}</span>'
        for el in elements
    )
    body = f"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  #map {{ width: 100%; height: calc(100vh - 230px); min-height: 520px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
  .filters {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-top: 14px; }}
  select {{ height: 34px; border: 1px solid var(--line); border-radius: 6px; padding: 0 8px; background: #fff; }}
</style>
<header>
  <h1>World Map: REE Mineral Distribution</h1>
  <p>Each point is one mineral locality record. Color shows the queried REE element; popup links go to Mindat locality pages.</p>
  <div class="legend">{legend}</div>
  <div class="filters">
    <label>Element <select id="elementFilter"><option value="all">All elements</option>{''.join(f'<option value="{html.escape(e)}">{html.escape(e)}</option>' for e in elements)}</select></label>
    <label>Country <select id="countryFilter"><option value="all">All countries</option>{''.join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in countries)}</select></label>
    <span id="countLabel"></span>
  </div>
</header>
<div id="map"></div>
<main><p><a href="index.html">Back to dashboard</a></p></main>
"""
    scripts = f"""
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const points = {json.dumps(data, ensure_ascii=False)};
const colors = {json.dumps(ELEMENT_COLORS)};
const map = L.map('map', {{ worldCopyJump: true }}).setView([20, 35], 2);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 8,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);
let layer = L.layerGroup().addTo(map);
function render() {{
  layer.clearLayers();
  const element = document.getElementById('elementFilter').value;
  const country = document.getElementById('countryFilter').value;
  const filtered = points.filter(p => (element === 'all' || p.element === element) && (country === 'all' || p.country === country));
  filtered.forEach(p => {{
    L.circleMarker([p.lat, p.lon], {{
      radius: 5,
      color: colors[p.element] || '#334155',
      weight: 1,
      fillOpacity: 0.72
    }}).bindPopup(`<strong>${{p.name}}</strong><br>Element: ${{p.element}} (${{p.element_name}})<br>REE group: ${{p.group}}<br>Country: ${{p.site_country}}<br><a href="${{p.url}}" target="_blank" rel="noreferrer">Mindat page</a>`).addTo(layer);
  }});
  document.getElementById('countLabel').textContent = `${{filtered.length}} records shown`;
}}
document.getElementById('elementFilter').addEventListener('change', render);
document.getElementById('countryFilter').addEventListener('change', render);
render();
</script>
"""
    (OUT / "world_mineral_distribution_map.html").write_text(page("World REE Mineral Distribution", body, scripts), encoding="utf-8")


def scale_points(items: list[dict], width: int, height: int, pad: int) -> list[tuple[float, float]]:
    lons = [x["longitude"] for x in items]
    lats = [x["latitude"] for x in items]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    points = []
    for item in items:
        x = pad + (item["longitude"] - min_lon) / (max_lon - min_lon or 1) * (width - 2 * pad)
        y = height - pad - (item["latitude"] - min_lat) / (max_lat - min_lat or 1) * (height - 2 * pad)
        points.append((x, y))
    return points


def scatter_svg(localities: list[dict], width: int = 960, height: int = 520) -> str:
    points = scale_points(localities, width, height, 48)
    circles = []
    for loc, (x, y) in zip(localities, points):
        cluster = int(loc["cluster"])
        color = PALETTE[cluster % len(PALETTE)]
        title = html.escape(
            f"{loc['locality_name']} | cluster {cluster} | {loc['country']} | elements: {', '.join(sorted(loc['query_elements']))}"
        )
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" opacity="0.72" '
            f'data-tip="{title}"></circle>'
        )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Cluster scatter by longitude and latitude">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
  <line x1="48" y1="{height-48}" x2="{width-28}" y2="{height-48}" stroke="#94a3b8"></line>
  <line x1="48" y1="28" x2="48" y2="{height-48}" stroke="#94a3b8"></line>
  <text x="{width/2}" y="{height-12}" text-anchor="middle" class="axis-label">Longitude</text>
  <text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" class="axis-label">Latitude</text>
  {''.join(circles)}
</svg>
"""


def write_cluster_results(localities: list[dict], cluster_info: dict) -> None:
    counts = Counter(loc["cluster"] for loc in localities)
    legend = "".join(
        f'<span><i class="swatch" style="background:{PALETTE[c % len(PALETTE)]}"></i>Cluster {c}: {counts[c]}</span>'
        for c in sorted(counts)
    )
    data = [
        {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "cluster": loc["cluster"],
            "name": loc["locality_name"],
            "country": loc["country"],
            "elements": ", ".join(sorted(loc["query_elements"])),
            "url": loc["mindat_locality_url"],
        }
        for loc in localities
    ]
    body = f"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>#clusterMap {{ width: 100%; height: 500px; border: 1px solid var(--line); border-radius: 8px; }}</style>
<header>
  <h1>Clustering Results</h1>
  <p>KMeans selected k={cluster_info['best_k']} using the highest silhouette score from k=3 to k=10. Features combine latitude, longitude, and REE element presence flags.</p>
  <div class="metric-row">
    <div class="metric"><strong>{len(localities)}</strong>clustered localities</div>
    <div class="metric"><strong>{cluster_info['best_k']}</strong>clusters</div>
    <div class="metric"><strong>{cluster_info['best_silhouette']:.3f}</strong>silhouette</div>
  </div>
  <div class="legend">{legend}</div>
</header>
<main>
  <section class="panel">
    <h2>Cluster Map</h2>
    <div id="clusterMap"></div>
  </section>
  <section class="panel">
    <h2>Cluster Scatter: Longitude vs Latitude</h2>
    {scatter_svg(localities)}
  </section>
  <p><a href="index.html">Back to dashboard</a></p>
</main>
<div id="tooltip" class="tooltip"></div>
"""
    scripts = f"""
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const points = {json.dumps(data, ensure_ascii=False)};
const palette = {json.dumps(PALETTE)};
const map = L.map('clusterMap', {{ worldCopyJump: true }}).setView([20, 35], 2);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 8,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);
points.forEach(p => {{
  L.circleMarker([p.lat, p.lon], {{
    radius: 5,
    color: palette[p.cluster % palette.length],
    weight: 1,
    fillOpacity: 0.75
  }}).bindPopup(`<strong>${{p.name}}</strong><br>Cluster: ${{p.cluster}}<br>Country: ${{p.country}}<br>Elements: ${{p.elements}}<br><a href="${{p.url}}" target="_blank" rel="noreferrer">Mindat page</a>`).addTo(map);
}});
const tooltip = document.getElementById('tooltip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', event => {{
    tooltip.textContent = el.dataset.tip;
    tooltip.style.left = `${{event.clientX + 12}}px`;
    tooltip.style.top = `${{event.clientY + 12}}px`;
    tooltip.style.opacity = 1;
  }});
  el.addEventListener('mouseleave', () => tooltip.style.opacity = 0);
}});
</script>
"""
    (OUT / "cluster_results.html").write_text(page("REE Clustering Results", body, scripts), encoding="utf-8")


def bar_svg(counter: Counter, title: str, color: str = "#2563eb", width: int = 720, row_h: int = 30) -> str:
    items = counter.most_common()
    height = 44 + row_h * len(items)
    max_value = max(counter.values()) if counter else 1
    rows = []
    for i, (label, value) in enumerate(items):
        y = 34 + i * row_h
        bar_w = (width - 210) * value / max_value
        rows.append(
            f'<text x="8" y="{y + 16}" class="bar-label">{html.escape(str(label))}</text>'
            f'<rect x="150" y="{y}" width="{bar_w:.1f}" height="19" fill="{color}" opacity="0.78"></rect>'
            f'<text x="{158 + bar_w:.1f}" y="{y + 15}" class="bar-label">{value}</text>'
        )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
  <text x="8" y="20" font-size="15" font-weight="700" fill="#1e293b">{html.escape(title)}</text>
  {''.join(rows)}
</svg>
"""


def write_auxiliary_charts(records: list[dict], localities: list[dict]) -> None:
    cluster_counts = Counter(loc["cluster"] for loc in localities)
    element_counts = Counter(r["query_element"] for r in records)
    country_counts = Counter(r["query_country"] for r in records)
    group_counts = Counter(r["ree_group"] for r in records)

    by_cluster_elements: dict[int, Counter] = defaultdict(Counter)
    for loc in localities:
        for element in loc["query_elements"]:
            by_cluster_elements[loc["cluster"]][element] += 1

    cluster_element_rows = []
    elements = sorted(element_counts)
    for cluster in sorted(cluster_counts):
        total = sum(by_cluster_elements[cluster].values()) or 1
        cells = "".join(
            f'<td style="background:rgba(37,99,235,{by_cluster_elements[cluster][el] / total:.2f})">{by_cluster_elements[cluster][el]}</td>'
            for el in elements
        )
        cluster_element_rows.append(f"<tr><th>Cluster {cluster}</th>{cells}</tr>")

    table = f"""
<table>
  <thead><tr><th></th>{''.join(f'<th>{html.escape(el)}</th>' for el in elements)}</tr></thead>
  <tbody>{''.join(cluster_element_rows)}</tbody>
</table>
"""
    body = f"""
<style>
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid var(--line); padding: 8px; text-align: right; font-size: 13px; }}
  th:first-child, td:first-child {{ text-align: left; }}
</style>
<header>
  <h1>Supporting Charts</h1>
  <p>Simple summary views for explaining the clustering and mineral composition.</p>
</header>
<main>
  <section class="grid">
    <div class="panel">{bar_svg(cluster_counts, "Localities per Cluster", "#2563eb")}</div>
    <div class="panel">{bar_svg(element_counts, "Records per Mineral Element", "#f97316")}</div>
    <div class="panel">{bar_svg(country_counts, "Records per Reserve Country", "#16a34a")}</div>
    <div class="panel">{bar_svg(group_counts, "Records per REE Group", "#9333ea")}</div>
  </section>
  <section class="panel">
    <h2>Cluster by Mineral Element Count</h2>
    {table}
  </section>
  <p><a href="index.html">Back to dashboard</a></p>
</main>
"""
    (OUT / "auxiliary_charts.html").write_text(page("REE Supporting Charts", body), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, localities = load_data()
    symbols = load_ree_symbols()
    cluster_info = cluster_localities(localities, symbols)
    write_index(records, localities, cluster_info)
    write_world_map(records)
    write_cluster_results(localities, cluster_info)
    write_auxiliary_charts(records, localities)
    print(f"Generated visualizations in {OUT}")
    print(f"Cluster file: {CLUSTER_CSV}")
    print(f"Best k: {cluster_info['best_k']} silhouette: {cluster_info['best_silhouette']:.3f}")


if __name__ == "__main__":
    main()
