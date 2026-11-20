from math import comb

import numpy as np
import pandas as pd

from model import DATA, RESULTS, distances


def uniform_nearest(ordered, removed):
    m = len(ordered)
    counts = [comb(m - j - 1, removed - j) for j in range(removed + 1)]
    assert sum(counts) == comb(m, removed)
    return np.average(ordered[:removed + 1], weights=counts)


def measure(area, paired):
    panel = pd.read_csv(DATA / f'{area}_panel.csv', index_col='date').loc['2023':]
    sites = pd.read_csv(DATA / f'{area}_stations.csv', index_col='site').loc[panel.columns]
    km = pd.DataFrame(distances(sites), index=sites.index, columns=sites.index)
    reporting = {day: row.dropna().index for day, row in panel.iterrows()}
    groups = pd.read_csv(RESULTS / f'{area}_review_footprints.csv')
    members = groups.groupby('footprint').site.agg(set).to_dict()
    cases = pd.read_csv(RESULTS / f'{area}_review_footprint_predictions.csv.gz')
    cases = cases[cases.date >= '2023-01-01']
    data = []
    for (day, site), matches in cases.groupby(['date', 'site'], sort=False):
        pool = km.loc[site, reporting[day].drop(site)].sort_values()
        uniform = {r: uniform_nearest(pool.to_numpy(), r) for r in matches.removed.unique()}
        for match in matches.itertuples(index=False):
            group = members[match.footprint]
            surviving = pool[~pool.index.isin(group)]
            assert site in group and len(surviving) == match.donors == len(pool) - match.removed
            assert np.isclose(surviving.iloc[0], match.distance_km)
        data.append((day, site, matches.removed.map(uniform).mean(),
                     matches.distance_km.mean(), pool.iloc[matches.removed].mean()))
    support = pd.DataFrame(data, columns=['date', 'site', 'random', 'grouped', 'nearest'])
    delta = support.grouped - support.random
    assert np.all(support.grouped <= support.nearest + 1e-9)
    errors = paired[paired.metro.eq(area)]
    assert len(errors) == 2 * len(support)
    assert set(zip(errors.date, errors.site)) == set(zip(support.date, support.site))
    by_site = errors.assign(delta=errors.footprint - errors.random).groupby(['method', 'site']).delta.mean()
    return {'metro': area, 'site_days': len(support), 'sites': support.site.nunique(),
            'median_random_km': support.random.median(), 'median_grouped_km': support.grouped.median(),
            'median_nearest_km': support.nearest.median(), 'median_paired_group_increment_km': delta.median(),
            'pct_group_farther_than_random_expectation': 100 * delta.gt(1e-9).mean(),
            'sites_positive_covariance': by_site.loc['covariance'].gt(0).sum(),
            'sites_positive_fixed_idw': by_site.loc['fixed_idw'].gt(0).sum()}


if __name__ == '__main__':
    pairs = pd.read_csv(RESULTS / 'review_geometry_pairs.csv.gz')
    paired = pairs[pairs.design.eq('footprint') & pairs.method.isin(['covariance', 'fixed_idw'])]
    summary = pd.DataFrame([measure(area, paired) for area in ['LA', 'NY', 'CH']])
    summary.to_csv(RESULTS / 'review_spatial_support.csv', index=False, float_format='%.6f')
    print(summary.to_string(index=False))
