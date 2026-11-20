from itertools import combinations, product
from math import comb

import numpy as np
import pandas as pd

from download import REGIONS
from model import DATA, RESULTS, distances, fit

CONFIGS = list(product([1, 2, 3], [3, 5, 8, 99]))
METHODS = ['fixed_idw', 'tuned_idw', 'covariance']
MASSES = [7, 14, 28]


def random_donors(pool, loss, rng, draws=32):
    if comb(len(pool), loss) <= draws:
        dropped = np.asarray(list(combinations(range(len(pool)), loss)))
        exhaustive = True
    else:
        u = rng.random((draws, len(pool)))
        dropped = np.argpartition(u, loss - 1, axis=1)[:, :loss]
        exhaustive = False
    keep = np.ones((len(dropped), len(pool)), dtype=bool)
    np.put_along_axis(keep, dropped, False, axis=1)
    donor_rows = np.broadcast_to(pool, keep.shape)[keep].reshape(len(dropped), -1)
    return donor_rows, len(donor_rows) if exhaustive else draws // 2


def weights(km, target, donors, exponent):
    return np.maximum(km[target, donors], 1.0) ** (-float(exponent))


def tune(panel, km, neighbor_order):
    rng = np.random.default_rng(3029)
    totals = np.zeros(len(CONFIGS))
    count = 0
    for day, row in panel.loc['2019':'2020'].iterrows():
        values = row.to_numpy()
        reporting = np.isfinite(values)
        if reporting.sum() < 9:
            continue
        for target in np.flatnonzero(reporting):
            ordered = neighbor_order[target]
            pool = ordered[reporting[ordered] & (ordered != target)]
            drop = len(pool) * 30 // 100
            random, _ = random_donors(pool, drop, rng, draws=8)
            donors = np.vstack([pool[drop:], random])
            for j, (exponent, cap) in enumerate(CONFIGS):
                use = donors[:, :cap]
                w = weights(km, target, use, exponent)
                pred = (values[use] * w).sum(axis=1) / w.sum(axis=1)
                loss = abs(pred - values[target])
                totals[j] += (loss[0] + loss[1:].mean()) / 2
            count += 1
    best = int(np.argmin(totals))
    grid = pd.DataFrame(CONFIGS, columns=['power', 'k'])
    grid['mean_mae'] = totals / count
    grid['site_days'] = count
    grid['selected'] = grid.index == best
    return CONFIGS[best], grid


def estimates(donors, target, values, mu, cov, km, config):
    x = values[donors]
    d = km[target, donors]
    w = weights(km, target, donors, 2)
    raw = (x * w).sum(axis=1) / w.sum(axis=1)
    exponent, cap = config
    wt = weights(km, target, donors[:, :cap], exponent)
    tuned = (x[:, :cap] * wt).sum(axis=1) / wt.sum(axis=1)
    cross = cov[target, donors]
    blocks = cov[donors[:, :, None], donors[:, None, :]]
    coef = np.linalg.solve(blocks, cross[..., None])[..., 0]
    empirical = mu[target] + (coef * (x - mu[donors])).sum(axis=1)
    var = np.maximum(cov[target, target] - (coef * cross).sum(axis=1), 1e-5)
    return np.column_stack([raw, tuned, empirical, d.min(axis=1), np.sqrt(var), x.std(axis=1)])


def score(city):
    panel = pd.read_csv(DATA / f'{city}_panel.csv', index_col='date', parse_dates=True)
    sites = pd.read_csv(DATA / f'{city}_stations.csv', index_col='site').loc[panel.columns]
    km = distances(sites)
    order = np.argsort(km, axis=1, kind='stable')
    means, cov, _, _ = fit(panel.loc[:'2020'], km)
    config, grid = tune(panel, km, order)
    grid.assign(metro=city).to_csv(RESULTS / f'{city}_review_idw_tuning.csv', index=False)
    k = len(sites) * 30 // 100
    footprints = sorted({tuple(sorted((i, *order[i][order[i] != i][:k - 1])))
                         for i in range(len(sites))})
    membership = np.zeros((len(footprints), len(sites)), dtype=bool)
    for j, f in enumerate(footprints):
        membership[j, list(f)] = True
    pd.DataFrame([(j, panel.columns[i]) for j, f in enumerate(footprints) for i in f],
                 columns=['footprint', 'site']).to_csv(RESULTS / f'{city}_review_footprints.csv', index=False)

    rng = np.random.default_rng(8671)
    comparisons, case_predictions = [], []
    for day, row in panel.loc['2021':].iterrows():
        values = row.to_numpy()
        mu = means[day.month - 1]
        reporting = np.isfinite(values)
        if reporting.sum() < 9:
            continue
        for target in np.flatnonzero(reporting):
            site, y = panel.columns[target], values[target]
            ordered = order[target]
            pool = ordered[reporting[ordered] & (ordered != target)]
            date_cache = {}

            def paired_losses(loss):
                if loss not in date_cache:
                    donors, half = random_donors(pool, loss, rng)
                    predictions = estimates(donors, target, values, mu, cov, km, config)
                    near = estimates(pool[loss:][None, :], target, values, mu, cov, km, config)[0]
                    error = abs(predictions[:, :3] - y)
                    date_cache[loss] = (error.mean(axis=0), abs(near[:3] - y), error[:half].mean(axis=0))
                return date_cache[loss]

            if day.year >= 2023:
                loss = len(pool) * 30 // 100
                uniform, nearest, half = paired_losses(loss)
                for j, name in enumerate(METHODS):
                    comparisons.append((day, site, 'target_nearest', name, 1, loss,
                                        uniform[j], nearest[j], np.nan, half[j]))
            matched = []
            for j in np.flatnonzero(membership[:, target]):
                available = pool[~membership[j, pool]]
                loss = len(pool) - len(available)
                if not loss:
                    continue
                assert len(available) >= 3
                result = estimates(available[None, :], target, values, mu, cov, km, config)[0]
                case_predictions.append((day, site, j, y, loss, len(available), *result))
                if day.year >= 2023:
                    uniform, nearest, half = paired_losses(loss)
                    matched.append([uniform, nearest, abs(result[:3] - y), half])
            if matched:
                errors = np.mean(matched, axis=0)
                removed = len(pool) - (~membership[np.flatnonzero(membership[:, target])][:, pool]).sum(axis=1)
                for j, name in enumerate(METHODS):
                    comparisons.append((day, site, 'footprint', name, len(matched), removed[removed > 0].mean(),
                                        *errors[:, j]))
    columns = ['date', 'site', 'design', 'method', 'n_footprints', 'mean_removed',
               'random', 'nearest', 'footprint', 'random_half']
    pair = pd.DataFrame(comparisons, columns=columns).assign(metro=city)
    fields = ['date', 'site', 'footprint', 'y', 'removed', 'donors', *METHODS,
              'distance_km', 'conditional_sd', 'donor_sd']
    fp = pd.DataFrame(case_predictions, columns=fields).rename(columns={'covariance': 'empirical'})
    fp.to_csv(RESULTS / f'{city}_review_footprint_predictions.csv.gz', index=False)
    print(city, len(footprints), len(fp), config, flush=True)
    return pair


def date_bootstrap(values, dates, block):
    daily = pd.DataFrame({'delta': values, 'n': 1}, index=dates).groupby(level=0).sum()
    rng = np.random.default_rng(1849)
    sums = np.zeros((2000, 2))
    for year in sorted(daily.index.year.unique()):
        calendar = pd.date_range(f'{year}-01-01', f'{year}-12-31')
        data = daily.reindex(calendar, fill_value=0).to_numpy()
        n = len(calendar)
        start = rng.integers(n, size=(len(sums), (n + block - 1) // block))
        draws = ((start[..., None] + np.arange(block)) % n).reshape(len(sums), -1)[:, :n]
        sums += data[draws].sum(axis=1)
    assert (sums[:, 1] > 0).all()
    return np.quantile(sums[:, 0] / sums[:, 1], [0.025, 0.975])


def summarize(pairs):
    rows, yearly = [], []
    for keys, group in pairs.groupby(['metro', 'design', 'method']):
        for pattern, reference in [('nearest', 'random'), ('footprint', 'random'),
                                   ('nearest', 'footprint')]:
            if group[pattern].isna().all() or group[reference].isna().all():
                continue
            delta = group[pattern] - group[reference]
            half = 'random_half' if reference == 'random' else reference
            delta_half = group[pattern] - group[half]
            ticks = pd.DatetimeIndex(group.date)
            bands = np.concatenate([date_bootstrap(delta.to_numpy(), ticks, l) for l in MASSES])
            rows.append((*keys, pattern, reference, len(group), group.date.nunique(), group.site.nunique(),
                         group.n_footprints.sum(), group.mean_removed.mean(), group[reference].mean(),
                         group[pattern].mean(), delta.mean(), *bands,
                         delta.groupby(group.date).mean().mean(), delta.groupby(group.site).mean().mean(),
                         delta_half.mean()))
            for year, y in delta.groupby(ticks.year):
                yearly.append((*keys, pattern, reference, year, len(y), y.mean()))
    columns = ['metro', 'design', 'method', 'pattern', 'reference', 'site_days', 'dates', 'sites', 'site_date_masks',
               'mean_removed', 'reference_mae', 'stress_mae', 'delta_mae', 'lo7', 'hi7', 'lo14', 'hi14',
               'lo28', 'hi28', 'date_balanced', 'site_balanced', 'delta_half_draws']
    pd.DataFrame(rows, columns=columns).to_csv(RESULTS / 'review_geometry_summary.csv', index=False)
    pd.DataFrame(yearly, columns=['metro', 'design', 'method', 'pattern', 'reference', 'year', 'site_days',
                                  'delta_mae']).to_csv(RESULTS / 'review_geometry_by_year.csv', index=False)


if __name__ == '__main__':
    pairs = pd.concat([score(city) for city in REGIONS], ignore_index=True)
    pairs.to_csv(RESULTS / 'review_geometry_pairs.csv.gz', index=False)
    summarize(pairs)
