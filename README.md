# Daily fine particulate matter reconstruction

This repository contains the analysis and statistical figure scripts for *Daily PM₂.₅ reconstruction, interval calibration and threshold screening under grouped monitor loss in three U.S. networks*. The networks cover the South Coast Air Basin, metropolitan New York and metropolitan Chicago. The evaluation compares group, nearest and uniform-random monitor removal at matched donor counts. Historical point models use 2017–2020 observations, interval quantiles use 2021–2022 residuals, and the final evaluation uses 2023–2025 target-days.

## Run the analysis

Run these commands from the repository root with Python 3.12. The downloader fetches EPA AQS daily summaries for parameter 88101 and the CARB South Coast basin boundary. To replay the original analysis with the archived inputs, place the 36 AQS query snapshots and `south_coast.geojson` in `data/raw/` before running the downloader. The downloader skips files that are already present. A fresh download can reflect later provider revisions. `prepare.py` records input row counts, acquisition times and snapshot checksums in `data/sources.csv`.

```text
python -m pip install -r code/requirements.txt
python code/download.py
python code/prepare.py
python code/model.py LA
python code/model.py NY
python code/model.py CH
python code/assess.py
python code/review_geometry.py
python code/review_intervals.py
python code/review_spatial_support.py
python code/transfer_validation.py
python code/expanded_checks.py
python code/referee_geometry.py
python code/referee_screening.py
python code/referee_support.py
python code/plot_statistical_figure.py
python code/plot_expanded_figures.py
python code/plot_referee_figures.py
```

## Main outputs

| File in `results/` | Contents |
| --- | --- |
| `review_geometry_summary.csv` | Paired reconstruction error and date-block intervals |
| `distance_conditioned.csv`, `group_strength.csv` | Closest-donor matching and group-size sensitivity |
| `review_interval_summary.csv`, `interval_score_components.csv` | Within-network calibration and interval-score decomposition |
| `CH_transfer_intervals.csv`, `transfer_single_source.csv` | Pooled and single-source interval transfer to Chicago |
| `screening_confusion.csv`, `screening_budget_summary.csv` | Three-way screening and fixed-budget review outcomes at 25 and 35 µg/m³ |
| `screening_schedules.csv` | Integer review counts from the simulated shared-group schedules and target-nearest comparison |

The plotting scripts write the workflow diagram and nine statistical figures as PDF and PNG files under `manuscript/figures/`. Source snapshots and generated outputs are local to each run.

Data providers. [U.S. EPA AQS API](https://aqs.epa.gov/aqsweb/documents/data_api.html) and [CARB Air Basins GIS](https://gis.carb.ca.gov/hosting/rest/services/California_Air_Resources_Board_Air_Basin_MIL1/MapServer/0).
