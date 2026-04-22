# IDA Donor Readiness Index — Streamlit Dashboard

Interactive dashboard for the IDA Sovereign Donor Readiness capstone project.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Folder structure
Place this file (`app.py`) at the root of the project. The dashboard expects:

```
app.py
requirements.txt
data/
  outputs/
    dri_output.csv             ← from main.py
    heckman_diagnostics.txt    ← from main.py
  processed/
    alignment_scores.csv       ← from main.py
    master.csv                 ← from main.py
    capacity_scores.csv        ← from main.py
```

If you are running this from a different directory, update the `BASE` path
at the top of `app.py`:

```python
BASE = Path("/path/to/your/project")
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### 4. Running in Google Colab
In a Colab cell, run:

```python
# Install dependencies
!pip install streamlit pyngrok -q

# Start the tunnel
from pyngrok import ngrok
import subprocess, threading, time

# Update BASE path in app.py first if needed
proc = subprocess.Popen(["streamlit", "run", "app.py",
                         "--server.port", "8501",
                         "--server.headless", "true"])
time.sleep(3)
tunnel = ngrok.connect(8501)
print("Dashboard URL:", tunnel.public_url)
```

## Pages

| Page | Description |
|---|---|
| Overview | KPI summary, segment breakdown, top 10 gap chart |
| Country Explorer | Drill-down profile for any individual country |
| Gap Analysis | Gap ranking, giving rate chart, capacity scatter |
| Prospect Ranking | Filterable ranked table + engagement priority matrix |
| World Map | Interactive choropleth (dark red = under-contributor, dark blue = over-contributor) |
| Model Diagnostics | Heckman coefficient comparison, VIF table, full diagnostics text |

## Sidebar Filters

All pages (except Model Diagnostics) respect the global filters in the sidebar:
- **Donor Segment** — filter by any combination of the 5 segments
- **Income Group** — HIC / UMC / LMC / LIC
- **Min. GDP** — exclude very small economies

## Notes

- The World Map uses the same symlog colour scale as `report.py` so it is
  consistent with the static HTML map output from the main pipeline.
- The `Country Explorer` page always shows the full dataset (not filtered)
  so you can look up any country regardless of sidebar settings.
- Download buttons on the Prospect Ranking page export the current filtered
  view as CSV.
