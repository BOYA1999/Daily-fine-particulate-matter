import json

import numpy as np
import pandas as pd

from download import REGIONS
from model import DATA, RESULTS, distances
from review_intervals import predict_bounds, scales


def year_support():
    rows = []
    for city in REGIONS:
        panel = pd.read_csv(DATA / f'{city}_panel.csv', index_col='date', parse_dates=['date'])
        for year, values in panel.groupby(panel.index.year):
            counts = values.notna().sum(axis=1)
            eligible = counts.ge(9)
            rows.append(dict(metro=city, year=year, calendar_days=len(values),
                             sites_reporting=int(values.notna().any().sum()),
                             sites_eligible=int(values.loc[eligible].notna().any().sum()),
                             reporting_site_days=int(counts.sum()), eligible_dates=int(eligible.sum()),
                             eligible_site_days=int(counts.loc[eligible].sum())))
    pd.DataFrame(rows).to_csv(RESULTS / 'year_support.csv', index=False)


def station_errors():
    pairs = pd.read_csv(RESULTS / 'review_geometry_pairs.csv.gz')
    pairs = pairs.query('design == "footprint" and method in ["covariance", "fixed_idw"]')
    paired = pairs.assign(delta=pairs.footprint - pairs.random)
    summary = paired.groupby(['metro', 'method', 'site']).agg(
        site_days=('date', 'size'), random_mae=('random', 'mean'),
        grouped_mae=('footprint', 'mean'), delta_mae=('delta', 'mean'))
    summary.to_csv(RESULTS / 'station_error_profiles.csv')


def interval_profiles():
    original = pd.read_csv(RESULTS / 'review_interval_summary.csv').query(
        'center == "covariance" and calibration == "footprint" and test_design == "footprint" and threshold == 35'
    ).set_index(['metro', 'scale'])
    results = []
    for city in REGIONS:
        dev = pd.read_csv(RESULTS / f'{city}_development.csv.gz')
        footprint = pd.read_csv(RESULTS / f'{city}_review_footprint_predictions.csv.gz', parse_dates=['date'])
        cases = predict_bounds({'footprint': footprint}, {'footprint': scales(dev, footprint)},
                              'footprint', 'footprint')
        for method in ('constant', 'combined'):
            lo, hi = cases[method + '_lo'], cases[method + '_hi']
            y = cases.y
            joined = cases.assign(coverage=(lo <= y) & (y <= hi), upper_miss=y > hi,
                                  width=hi-lo, score=hi-lo+20*((lo-y).clip(lower=0)+(y-hi).clip(lower=0)))
            onsite = joined.groupby(['date', 'site'], as_index=False).agg(
                y=('y', 'first'), coverage=('coverage', 'mean'), upper_miss=('upper_miss', 'mean'),
                width=('width', 'mean'), score=('score', 'mean'))
            baseline = original.loc[city, method]
            assert len(onsite) == round(baseline.site_days)
            assert np.allclose(onsite[['coverage', 'upper_miss', 'width', 'score']].mean(),
                               baseline[['coverage', 'upper_miss', 'width', 'interval_score']].astype(float))
            onsite['observed_band'] = pd.cut(onsite.y, [-np.inf, 25, 35, np.inf],
                                              labels=['le25', '25to35', 'gt35'])
            onsite['test_year'] = onsite.date.dt.year
            for kind in ('observed_band', 'test_year'):
                profile = onsite.groupby(kind, observed=True).agg(
                    site_days=('y', 'size'), coverage=('coverage', 'mean'),
                    upper_miss=('upper_miss', 'mean'), width=('width', 'mean'),
                    score=('score', 'mean')).reset_index(names='stratum')
                results.append(profile.assign(metro=city, scale=method, slice_type=kind))
    pd.concat(results, ignore_index=True).to_csv(RESULTS / 'interval_profiles.csv', index=False)


def mask_example():
    city = 'CH'
    panel = pd.read_csv(DATA / f'{city}_panel.csv', index_col='date')
    stations = pd.read_csv(DATA / f'{city}_stations.csv', index_col='site').loc[panel.columns]
    km = pd.DataFrame(distances(stations), index=stations.index, columns=stations.index)
    groups = pd.read_csv(RESULTS / f'{city}_review_footprints.csv')
    members = groups.groupby('footprint').site.agg(set)
    archive = pd.read_csv(RESULTS / f'{city}_review_footprint_predictions.csv.gz')
    candidates = archive.query('date >= "2023-01-01" and removed == 3 and donors >= 9')
    for row in candidates.sort_values(['date', 'site', 'footprint']).itertuples(index=False):
        observed = set(panel.loc[row.date].dropna().index)
        original = observed - {row.site}
        shared = original & members.loc[row.footprint]
        ranked = km.loc[row.site, sorted(original)].sort_values(kind='stable')
        nearest = set(ranked.index[:row.removed])
        if shared != nearest:
            break
    assert shared != nearest and len(original) == row.removed + row.donors and len(shared) == row.removed
    rng = np.random.default_rng(418)
    random = set(rng.choice(sorted(original), size=row.removed, replace=False))
    features = []
    meta = dict(metro=city, date=row.date, target=row.site, group=int(row.footprint),
                original_donors=len(original), removed=int(row.removed), kept=int(row.donors), scenarios={})
    for scenario, withheld in [('random', random), ('shared', shared), ('nearest', nearest)]:
        d = float(ranked.drop(list(withheld)).min())
        meta['scenarios'][scenario] = {'removed_ids': sorted(withheld), 'closest_km': d}
        for site, position in stations.iterrows():
            status = ('target' if site == row.site else 'withheld' if site in withheld else
                      'kept' if site in original else 'absent')
            features.append({'type': 'Feature', 'properties': dict(site=site, scenario=scenario, status=status),
                             'geometry': {'type': 'Point', 'coordinates': [position.longitude, position.latitude]}})
    assert np.isclose(meta['scenarios']['shared']['closest_km'], row.distance_km)
    destination = DATA / 'map_sources' / 'mask_example.geojson'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}), encoding='utf-8')
    (RESULTS / 'mask_example.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    year_support()
    station_errors()
    interval_profiles()
    mask_example()
