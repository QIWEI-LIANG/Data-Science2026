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
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/source-sans-3@5.0.12/index.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/overlayscrollbars@2.10.1/styles/overlayscrollbars.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/admin-lte@4.0.0-rc7/dist/css/adminlte.min.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{ --ink:#172033; --muted:#5f6f85; --line:#d8e0eb; --panel:#fff; --bg:#f4f6f9; --accent:#0d6efd; }}
    body {{ font-family:"Source Sans 3", Arial, Helvetica, sans-serif; color:var(--ink); background:var(--bg); }}
    .app-main {{ background:var(--bg); }}
    .brand-link {{ text-decoration:none; }}
    .brand-image {{ width:34px; height:34px; display:inline-grid; place-items:center; border-radius:8px; background:#0d6efd; color:#fff; }}
    .app-content-header {{ border-bottom:1px solid var(--line); background:#fff; }}
    .page-summary {{ max-width:920px; color:var(--muted); }}
    .metric .inner p {{ color:#f8fafc; }}
    .controls-card .form-label {{ color:var(--muted); font-size:13px; font-weight:600; margin-bottom:4px; }}
    .view-actions .btn.active {{ color:#fff; background:#0d6efd; border-color:#0d6efd; }}
    #map {{ height:590px; min-height:58vh; border-radius:0 0 8px 8px; background:#eef3f8; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:8px 13px; font-size:13px; }}
    .legend span {{ display:inline-flex; gap:6px; align-items:center; }}
    .legend i {{ display:inline-block; width:11px; height:11px; border-radius:50%; }}
    .grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:16px; align-items:stretch; }}
    .chart-card {{ min-height:360px; }}
    .chart-card .card-body {{ display:flex; align-items:center; overflow:auto; }}
    .matrix-card .card-body {{ overflow:auto; }}
    svg {{ max-width:100%; height:auto; display:block; }}
    .bar-label {{ font-size:12px; fill:#334155; }}
    .matrix {{ width:100%; border-collapse:collapse; min-width:980px; }}
    .matrix th, .matrix td {{ border:1px solid var(--line); padding:8px; text-align:right; font-size:13px; }}
    .matrix th:first-child {{ text-align:left; position:sticky; left:0; background:#fff; z-index:1; }}
    .note {{ margin-top:10px; font-size:13px; color:var(--muted); }}
    .leaflet-popup-content {{ line-height:1.45; }}
    .popup-list {{ max-height:240px; overflow:auto; margin-top:8px; }}
    .popup-record {{ padding:7px 0; border-top:1px solid var(--line); }}
    .popup-record:first-child {{ border-top:0; }}
    @media (max-width:992px) {{ .grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:720px) {{ .app-content-header .container-fluid, .app-content .container-fluid {{ padding-left:12px; padding-right:12px; }} #map {{ height:500px; }} .chart-card {{ min-height:0; }} }}
  </style>
</head>
<body class="layout-fixed sidebar-expand-lg bg-body-tertiary">
  <div class="app-wrapper">
    <nav class="app-header navbar navbar-expand bg-body">
      <div class="container-fluid">
        <ul class="navbar-nav">
          <li class="nav-item">
            <a class="nav-link" data-lte-toggle="sidebar" href="#" role="button" aria-label="Toggle sidebar">
              <i class="bi bi-list"></i>
            </a>
          </li>
          <li class="nav-item d-none d-md-block"><a href="#mapPanel" class="nav-link">Map</a></li>
          <li class="nav-item d-none d-md-block"><a href="#matrixPanel" class="nav-link">Matrix</a></li>
          <li class="nav-item d-none d-md-block"><a href="#chartsPanel" class="nav-link">Charts</a></li>
        </ul>
        <ul class="navbar-nav ms-auto">
          <li class="nav-item">
            <span id="countLabel" class="badge text-bg-primary fs-6"></span>
          </li>
        </ul>
      </div>
    </nav>

    <aside class="app-sidebar bg-dark shadow" data-bs-theme="dark">
      <div class="sidebar-brand">
        <a href="#" class="brand-link">
          <span class="brand-image"><i class="bi bi-gem"></i></span>
          <span class="brand-text fw-light ms-2">REE Dashboard</span>
        </a>
      </div>
      <div class="sidebar-wrapper">
        <nav class="mt-2">
          <ul class="nav sidebar-menu flex-column" data-lte-toggle="treeview" role="menu" data-accordion="false">
            <li class="nav-item"><a href="#mapPanel" class="nav-link active"><i class="nav-icon bi bi-globe2"></i><p>Distribution Map</p></a></li>
            <li class="nav-item"><a href="#matrixPanel" class="nav-link"><i class="nav-icon bi bi-grid-3x3-gap"></i><p>Element Matrix</p></a></li>
            <li class="nav-item"><a href="#chartsPanel" class="nav-link"><i class="nav-icon bi bi-bar-chart"></i><p>Summary Charts</p></a></li>
          </ul>
        </nav>
      </div>
    </aside>

    <main class="app-main">
      <div class="app-content-header">
        <div class="container-fluid py-3">
          <div class="row align-items-center">
            <div class="col-lg-8">
              <h1 class="mb-1 fs-3">17 REE Elements Across the Top 5 Reserve Countries</h1>
              <p class="page-summary mb-0">Distribution of Mindat locality records for all 17 rare-earth elements in China, Brazil, Australia, Russia, and Vietnam.</p>
            </div>
            <div class="col-lg-4 mt-3 mt-lg-0">
              <ol class="breadcrumb float-lg-end mb-0">
                <li class="breadcrumb-item"><a href="#mapPanel">Home</a></li>
                <li class="breadcrumb-item active" aria-current="page">REE Analysis</li>
              </ol>
            </div>
          </div>
        </div>
      </div>

      <div class="app-content">
        <div class="container-fluid py-3">
          <div class="row g-3">
            <div class="col-12 col-sm-6 col-xl-3">
              <div class="small-box text-bg-primary metric">
                <div class="inner"><h3>{len(records)}</h3><p>Locality records</p></div>
                <i class="small-box-icon bi bi-pin-map"></i>
              </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
              <div class="small-box text-bg-success metric">
                <div class="inner"><h3>{len({row['site'] for row in records})}</h3><p>Named sites</p></div>
                <i class="small-box-icon bi bi-geo-alt"></i>
              </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
              <div class="small-box text-bg-warning metric">
                <div class="inner"><h3>{len(elements)}</h3><p>REE elements</p></div>
                <i class="small-box-icon bi bi-diagram-3"></i>
              </div>
            </div>
            <div class="col-12 col-sm-6 col-xl-3">
              <div class="small-box text-bg-danger metric">
                <div class="inner"><h3>{reserve_total:,}</h3><p>REO tonnes in top 5 reserves</p></div>
                <i class="small-box-icon bi bi-database"></i>
              </div>
            </div>
          </div>

          <div class="card controls-card mb-3">
            <div class="card-body">
              <div class="row g-3 align-items-end">
                <div class="col-12 col-md-4 col-xl-3">
                  <label class="form-label" for="elementFilter">Element</label>
                  <select class="form-select" id="elementFilter"><option value="all">All 17 elements</option>{element_options}</select>
                </div>
                <div class="col-12 col-md-4 col-xl-3">
                  <label class="form-label" for="countryFilter">Country</label>
                  <select class="form-select" id="countryFilter"><option value="all">All top 5 countries</option>{country_options}</select>
                </div>
                <div class="col-12 col-md-4 col-xl-6">
                  <div class="btn-group view-actions" role="group" aria-label="Map view controls">
                    <button id="globalView" class="btn btn-outline-primary active" type="button"><i class="bi bi-globe"></i> Global View</button>
                    <button id="asiaView" class="btn btn-outline-primary" type="button"><i class="bi bi-crosshair"></i> Asia Focus</button>
                    <button id="clusterToggle" class="btn btn-outline-secondary" type="button"><i class="bi bi-collection"></i> Cluster Mode</button>
                  </div>
                </div>
              </div>
              <div class="legend mt-3">{legend}</div>
            </div>
          </div>

          <div id="mapPanel" class="card mb-3">
            <div class="card-header">
              <h2 class="card-title mb-0">Interactive Locality Map</h2>
            </div>
            <div class="card-body p-0">
              <div id="map"></div>
            </div>
          </div>

          <div id="matrixPanel" class="card matrix-card mb-3">
            <div class="card-header">
              <h2 class="card-title mb-0">Country by Element Locality Matrix</h2>
            </div>
            <div class="card-body">
              {matrix}
              <p class="note">Counts are Mindat locality records in the processed dataset; blank/zero cells mean no valid locality records were found for that element-country pair.</p>
            </div>
          </div>

          <div id="chartsPanel" class="grid">
            <div class="card chart-card"><div class="card-body">{element_bar}</div></div>
            <div class="card chart-card"><div class="card-body">{country_bar}</div></div>
            <div class="card chart-card"><div class="card-body">{reserve_bar}</div></div>
            <div class="card chart-card"><div class="card-body">{group_bar}</div></div>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/overlayscrollbars@2.10.1/browser/overlayscrollbars.browser.es6.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/admin-lte@4.0.0-rc7/dist/js/adminlte.min.js"></script>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {json.dumps(records, ensure_ascii=False)};
    const clusters = {json.dumps(clusters, ensure_ascii=False)};
    const colors = {json.dumps(ELEMENT_COLORS)};
    const worldBounds = L.latLngBounds([[-85, -180], [85, 180]]);
    const map = L.map('map', {{
      preferCanvas: true,
      worldCopyJump: false,
      maxBounds: worldBounds,
      maxBoundsViscosity: 1.0,
      minZoom: 2
    }}).setView([22, 78], 3);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      noWrap: true,
      bounds: worldBounds,
      minZoom: 2,
      maxZoom: 9,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const pointLayer = L.layerGroup().addTo(map);
    let clusterMode = false;

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }}[ch]));
    }}

    function popupFor(p, isCluster=false) {{
      const countLine = isCluster ? `<br>Clustered localities: ${{p.count}}` : '';
      const link = p.url ? `<br><a href="${{esc(p.url)}}" target="_blank" rel="noreferrer">Mindat locality page</a>` : '';
      return `<strong>${{esc(isCluster ? `${{p.reserve_country}} - ${{p.element}} cluster` : p.site)}}</strong><br>` +
        `Element: ${{esc(p.element)}} (${{esc(p.element_name)}})<br>` +
        `REE group: ${{esc(p.group)}}<br>` +
        `Reserve country query: ${{esc(p.reserve_country)}}<br>` +
        `Site country: ${{esc(isCluster ? p.reserve_country : p.site_country)}}${{countLine}}${{link}}`;
    }}

    function popupForGroup(group) {{
      const items = group.items;
      const title = items.length === 1 ? esc(items[0].site) : `${{items.length}} records at this coordinate`;
      const rows = items.map(p => {{
        const link = p.url ? `<br><a href="${{esc(p.url)}}" target="_blank" rel="noreferrer">Mindat locality page</a>` : '';
        return `<div class="popup-record">` +
          `<strong>${{esc(p.site)}}</strong><br>` +
          `Element: ${{esc(p.element)}} (${{esc(p.element_name)}})<br>` +
          `REE group: ${{esc(p.group)}}<br>` +
          `Reserve country query: ${{esc(p.reserve_country)}}<br>` +
          `Site country: ${{esc(p.site_country)}}${{link}}` +
        `</div>`;
      }}).join('');
      return `<strong>${{title}}</strong><br>` +
        `Coordinate: ${{group.lat.toFixed(5)}}, ${{group.lon.toFixed(5)}}` +
        `<div class="popup-list">${{rows}}</div>`;
    }}

    function draw() {{
      pointLayer.clearLayers();
      const element = document.getElementById('elementFilter').value;
      const country = document.getElementById('countryFilter').value;
      let shown = 0;

      if (!clusterMode) {{
        const grouped = new Map();
        points.forEach(p => {{
          if ((element !== 'all' && p.element !== element) || (country !== 'all' && p.reserve_country !== country)) return;
          shown += 1;
          const key = `${{p.lat.toFixed(6)}},${{p.lon.toFixed(6)}}`;
          if (!grouped.has(key)) grouped.set(key, {{ lat: p.lat, lon: p.lon, items: [] }});
          grouped.get(key).items.push(p);
        }});
        grouped.forEach(group => {{
          const first = group.items[0];
          const radius = group.items.length > 1 ? Math.min(13, 5 + Math.sqrt(group.items.length) * 1.35) : 5;
          L.circleMarker([group.lat, group.lon], {{
            radius,
            color: '#ffffff',
            weight: 2,
            fillColor: colors[first.element] || '#475569',
            fillOpacity: 0.74,
            opacity: 1
          }}).bindPopup(popupForGroup(group), {{ maxWidth: 420 }}).addTo(pointLayer);
        }});
        document.getElementById('countLabel').textContent = `${{shown}} records shown`;
        return;
      }}

      clusters.forEach(p => {{
        if ((element !== 'all' && p.element !== element) || (country !== 'all' && p.reserve_country !== country)) return;
        shown += p.count;
        const radius = Math.min(18, 6 + Math.sqrt(p.count) * 1.25);
        L.circleMarker([p.lat, p.lon], {{
          radius,
          color: '#ffffff',
          weight: 2,
          fillColor: colors[p.element] || '#475569',
          fillOpacity: 0.82,
          opacity: 1
        }}).bindPopup(popupFor(p, true)).addTo(pointLayer);
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
