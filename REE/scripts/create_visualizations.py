from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path


try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        ROOT = cwd
    elif (cwd / "REE" / "data" / "processed").exists():
        ROOT = cwd / "REE"
    else:
        ROOT = Path(".").resolve()

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "visualizations"

LOCALITIES_CSV = PROCESSED / "top5_17ree_localities_analysis_ready.csv"
SUMMARY_CSV = PROCESSED / "top5_17ree_distribution_summary.csv"
COUNTRIES_CSV = PROCESSED / "top5_ree_reserve_countries.csv"
ELEMENTS_CSV = PROCESSED / "ree_17_elements.csv"
OUTPUT_HTML = OUT / "index.html"

ELEMENT_COLORS = {
    "Sc": "#4c78a8",
    "Y": "#72b7b2",
    "La": "#eeca3b",
    "Ce": "#f58518",
    "Pr": "#b279a2",
    "Nd": "#9c755f",
    "Pm": "#bab0ac",
    "Sm": "#ff9da6",
    "Eu": "#d37295",
    "Gd": "#54a24b",
    "Tb": "#59a14f",
    "Dy": "#8cd17d",
    "Ho": "#499894",
    "Er": "#86bcb6",
    "Tm": "#79706e",
    "Yb": "#9d755d",
    "Lu": "#af7aa1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_inputs() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    missing = [path for path in [LOCALITIES_CSV, SUMMARY_CSV, COUNTRIES_CSV, ELEMENTS_CSV] if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing processed input file(s):\n{formatted}")

    countries = read_csv(COUNTRIES_CSV)
    elements = read_csv(ELEMENTS_CSV)
    summary_rows = read_csv(SUMMARY_CSV)
    locality_rows = read_csv(LOCALITIES_CSV)

    element_order = {row["symbol"]: i for i, row in enumerate(elements)}
    country_order = {row["country"]: as_int(row["reserve_rank"]) for row in countries}

    records: list[dict] = []
    for row in locality_rows:
        lat = as_float(row["latitude"])
        lon = as_float(row["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if lat == 0 and lon == 0:
            continue
        records.append(
            {
                "lat": lat,
                "lon": lon,
                "element": row["query_element"],
                "element_name": row["element_name"],
                "group": row["ree_group"],
                "reserve_country": row["query_country"],
                "site_country": row["country"],
                "site": row["locality_name"],
                "url": row["mindat_locality_url"],
            }
        )

    summary_lookup = {
        (row["query_country"], row["query_element"]): as_int(row["locality_count"])
        for row in summary_rows
    }
    summary: list[dict] = []
    for country in sorted(countries, key=lambda row: as_int(row["reserve_rank"])):
        for element in sorted(elements, key=lambda row: element_order[row["symbol"]]):
            summary.append(
                {
                    "country": country["country"],
                    "element": element["symbol"],
                    "element_name": element["element_name"],
                    "group": element["ree_group"],
                    "locality_count": summary_lookup.get((country["country"], element["symbol"]), 0),
                    "reserves_reo_tonnes": as_int(country["reserves_reo_tonnes"]),
                }
            )

    return (
        sorted(countries, key=lambda row: country_order[row["country"]]),
        sorted(elements, key=lambda row: element_order[row["symbol"]]),
        summary,
        records,
    )


def make_matrix(countries: list[dict], elements: list[dict], summary: list[dict]) -> str:
    counts = {(row["country"], row["element"]): row["locality_count"] for row in summary}
    max_count = max((row["locality_count"] for row in summary), default=1) or 1
    headers = "".join(f"<th>{html.escape(row['symbol'])}</th>" for row in elements)
    rows = []
    for country in countries:
        cells = []
        for element in elements:
            count = counts.get((country["country"], element["symbol"]), 0)
            opacity = 0.08 + 0.82 * (count / max_count) if count else 0
            bg = f"rgba(37, 99, 235, {opacity:.2f})" if count else "#f8fafc"
            color = "#ffffff" if opacity > 0.45 else "#1e293b"
            cells.append(
                f'<td style="background:{bg}; color:{color}" '
                f'title="{html.escape(country["country"])} - {html.escape(element["symbol"])}: {count} localities">{count}</td>'
            )
        rows.append(f"<tr><th>{html.escape(country['country'])}</th>{''.join(cells)}</tr>")
    return f"""
<table class="matrix">
  <thead><tr><th>Country</th>{headers}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def bar_svg(items: list[tuple[str, int]], title: str, color: str) -> str:
    width = 760
    row_h = 28
    height = 46 + row_h * len(items)
    max_value = max((value for _, value in items), default=1) or 1
    rows = []
    for i, (label, value) in enumerate(items):
        y = 36 + i * row_h
        bar_w = (width - 250) * value / max_value
        rows.append(
            f'<text x="8" y="{y + 15}" class="bar-label">{html.escape(label)}</text>'
            f'<rect x="170" y="{y}" width="{bar_w:.1f}" height="18" fill="{color}" opacity="0.78"></rect>'
            f'<text x="{178 + bar_w:.1f}" y="{y + 14}" class="bar-label">{value}</text>'
        )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <rect width="{width}" height="{height}" fill="#ffffff"></rect>
  <text x="8" y="20" font-size="15" font-weight="700" fill="#1e293b">{html.escape(title)}</text>
  {''.join(rows)}
</svg>
"""


def build_clusters(records: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], dict] = {}
    for row in records:
        key = (row["reserve_country"], row["element"])
        item = buckets.setdefault(
            key,
            {
                "reserve_country": row["reserve_country"],
                "element": row["element"],
                "element_name": row["element_name"],
                "group": row["group"],
                "lat": 0.0,
                "lon": 0.0,
                "count": 0,
            },
        )
        item["lat"] += row["lat"]
        item["lon"] += row["lon"]
        item["count"] += 1
    clusters = []
    for item in buckets.values():
        clusters.append({**item, "lat": item["lat"] / item["count"], "lon": item["lon"] / item["count"]})
    return clusters


def write_html() -> None:
    countries, elements, summary, records = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)

    element_counts = Counter(row["element"] for row in records)
    country_counts = Counter(row["reserve_country"] for row in records)
    group_counts = Counter(row["group"] for row in records)
    reserve_total = sum(as_int(row["reserves_reo_tonnes"]) for row in countries)
    clusters = build_clusters(records)

    element_options = "".join(
        f'<option value="{html.escape(row["symbol"])}">{html.escape(row["symbol"])} - {html.escape(row["element_name"])}</option>'
        for row in elements
    )
    country_options = "".join(
        f'<option value="{html.escape(row["country"])}">{html.escape(row["country"])}</option>'
        for row in countries
    )
    legend = "".join(
        f'<span><i style="background:{ELEMENT_COLORS.get(row["symbol"], "#475569")}"></i>{html.escape(row["symbol"])}</span>'
        for row in elements
    )
    matrix = make_matrix(countries, elements, summary)
    element_bar = bar_svg([(row["symbol"], element_counts.get(row["symbol"], 0)) for row in elements], "Locality Records by REE Element", "#2563eb")
    country_bar = bar_svg([(row["country"], country_counts.get(row["country"], 0)) for row in countries], "Locality Records by Top Reserve Country", "#16a34a")
    reserve_bar = bar_svg([(row["country"], as_int(row["reserves_reo_tonnes"])) for row in countries], "REE Reserves by Country (REO tonnes)", "#f97316")
    group_bar = bar_svg([(label.upper(), group_counts[label]) for label in sorted(group_counts)], "Records by REE Group", "#9333ea")

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top 5 Countries REE Element Distribution</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{ --ink:#172033; --muted:#5f6f85; --line:#d8e0eb; --panel:#fff; --bg:#f5f7fb; --accent:#2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:22px 28px 14px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:26px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    p {{ margin:0 0 10px; color:var(--muted); line-height:1.5; }}
    main {{ padding:18px 28px 30px; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
    .metric {{ min-width:160px; background:#fff; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
    .metric strong {{ display:block; font-size:22px; color:var(--ink); }}
    .controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:14px; }}
    label {{ color:var(--muted); font-size:14px; }}
    select, button {{ height:34px; border:1px solid var(--line); border-radius:6px; padding:0 9px; background:#fff; color:var(--ink); }}
    select {{ margin-left:4px; }}
    button {{ cursor:pointer; }}
    button.active {{ border-color:var(--accent); background:#eff6ff; color:#1d4ed8; }}
    #map {{ height:560px; min-height:52vh; border-bottom:1px solid var(--line); background:#eef3f8; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:8px 13px; margin-top:12px; font-size:13px; }}
    .legend span {{ display:inline-flex; gap:6px; align-items:center; }}
    .legend i {{ display:inline-block; width:11px; height:11px; border-radius:50%; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; }}
    section {{ margin-top:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; overflow:auto; }}
    svg {{ max-width:100%; height:auto; display:block; }}
    .bar-label {{ font-size:12px; fill:#334155; }}
    .matrix {{ width:100%; border-collapse:collapse; min-width:980px; }}
    .matrix th, .matrix td {{ border:1px solid var(--line); padding:8px; text-align:right; font-size:13px; }}
    .matrix th:first-child {{ text-align:left; position:sticky; left:0; background:#fff; z-index:1; }}
    .note {{ margin-top:10px; font-size:13px; color:var(--muted); }}
    .leaflet-popup-content {{ line-height:1.45; }}
    @media (max-width:720px) {{ header, main {{ padding-left:16px; padding-right:16px; }} h1 {{ font-size:22px; }} #map {{ height:500px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>17 REE Elements Across the Top 5 Reserve Countries</h1>
    <p>Distribution of Mindat locality records for all 17 rare-earth elements in China, Brazil, Australia, Russia, and Vietnam.</p>
    <div class="metrics">
      <div class="metric"><strong>{len(records)}</strong>locality records</div>
      <div class="metric"><strong>{len({row['site'] for row in records})}</strong>named sites</div>
      <div class="metric"><strong>{len(elements)}</strong>REE elements</div>
      <div class="metric"><strong>{reserve_total:,}</strong>REO tonnes in top 5 reserves</div>
    </div>
    <div class="controls">
      <label>Element <select id="elementFilter"><option value="all">All 17 elements</option>{element_options}</select></label>
      <label>Country <select id="countryFilter"><option value="all">All top 5 countries</option>{country_options}</select></label>
      <button id="globalView" class="active" type="button">Global View</button>
      <button id="asiaView" type="button">Asia Focus</button>
      <button id="clusterToggle" type="button">Cluster Mode</button>
      <span id="countLabel"></span>
    </div>
    <div class="legend">{legend}</div>
  </header>

  <div id="map"></div>

  <main>
    <section>
      <h2>Country by Element Locality Matrix</h2>
      {matrix}
      <p class="note">Counts are Mindat locality records in the processed dataset; blank/zero cells mean no valid locality records were found for that element-country pair.</p>
    </section>
    <div class="grid">
      <section>{element_bar}</section>
      <section>{country_bar}</section>
      <section>{reserve_bar}</section>
      <section>{group_bar}</section>
    </div>
  </main>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {json.dumps(records, ensure_ascii=False)};
    const clusters = {json.dumps(clusters, ensure_ascii=False)};
    const colors = {json.dumps(ELEMENT_COLORS)};
    const map = L.map('map', {{ preferCanvas: true, worldCopyJump: true }}).setView([22, 78], 3);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 9,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const pointLayer = L.layerGroup().addTo(map);
    let clusterMode = false;

    function popupFor(p, isCluster=false) {{
      const countLine = isCluster ? `<br>Clustered localities: ${{p.count}}` : '';
      const link = p.url ? `<br><a href="${{p.url}}" target="_blank" rel="noreferrer">Mindat locality page</a>` : '';
      return `<strong>${{isCluster ? `${{p.reserve_country}} - ${{p.element}} cluster` : p.site}}</strong><br>` +
        `Element: ${{p.element}} (${{p.element_name}})<br>` +
        `REE group: ${{p.group}}<br>` +
        `Reserve country query: ${{p.reserve_country}}<br>` +
        `Site country: ${{isCluster ? p.reserve_country : p.site_country}}${{countLine}}${{link}}`;
    }}

    function draw() {{
      pointLayer.clearLayers();
      const element = document.getElementById('elementFilter').value;
      const country = document.getElementById('countryFilter').value;
      const source = clusterMode ? clusters : points;
      let shown = 0;
      source.forEach(p => {{
        if ((element !== 'all' && p.element !== element) || (country !== 'all' && p.reserve_country !== country)) return;
        shown += clusterMode ? p.count : 1;
        const radius = clusterMode ? Math.min(18, 6 + Math.sqrt(p.count) * 1.25) : 5;
        L.circleMarker([p.lat, p.lon], {{
          radius,
          color: '#ffffff',
          weight: 2,
          fillColor: colors[p.element] || '#475569',
          fillOpacity: clusterMode ? 0.82 : 0.72,
          opacity: 1
        }}).bindPopup(popupFor(p, clusterMode)).addTo(pointLayer);
      }});
      document.getElementById('countLabel').textContent = `${{shown}} records shown`;
    }}

    function setView(mode) {{
      if (mode === 'asia') {{
        map.setView([31, 103], 4);
        document.getElementById('asiaView').classList.add('active');
        document.getElementById('globalView').classList.remove('active');
      }} else {{
        map.setView([18, 60], 2);
        document.getElementById('globalView').classList.add('active');
        document.getElementById('asiaView').classList.remove('active');
      }}
    }}

    document.getElementById('elementFilter').addEventListener('change', draw);
    document.getElementById('countryFilter').addEventListener('change', draw);
    document.getElementById('globalView').addEventListener('click', () => setView('global'));
    document.getElementById('asiaView').addEventListener('click', () => setView('asia'));
    document.getElementById('clusterToggle').addEventListener('click', event => {{
      clusterMode = !clusterMode;
      event.currentTarget.classList.toggle('active', clusterMode);
      draw();
    }});
    draw();
  </script>
</body>
</html>
"""
    OUTPUT_HTML.write_text(page, encoding="utf-8")


def main() -> None:
    write_html()
    print(f"Generated single HTML interface: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
