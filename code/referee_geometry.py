from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from download import REGIONS
from model import DATA, RESULTS, distances, fit
from review_geometry import date_bootstrap
from review_intervals import case_weights, percentile

FITS = {
    'full': ('2017', '2020', {}),
    'early': ('2017', '2018', {}),
    'late': ('2019', '2020', {}),
    'overlap120': ('2017', '2020', {'min_overlap': 120}),
    'floor1e3': ('2017', '2020', {'eigen_floor': 1e-3}),
    'shrink01': ('2017', '2020', {'shrink': .01}),
    'shrink30': ('2017', '2020', {'shrink': .30}),
}
SIZES = (3, 6, 9)


def library(order, k):
    return sorted({tuple(sorted((i, *neighbors[neighbors != i][:k - 1])))
                   for i, neighbors in enumerate(order)})


def sampled_pool(pool, removed, dates, rng, rank=None, draws=32):
    if rank is None:
        fixed, tail, choose = np.array([], dtype=int), pool, removed
    else:
        fixed, tail, choose = pool[rank:rank + 1], pool[rank + 1:], removed - rank
    combinations_n = comb(len(tail), choose)
    if combinations_n <= draws:
        indices = np.array(list(combinations(range(len(tail)), choose)), dtype=int)
        indices = np.broadcast_to(indices, (dates, *indices.shape))
    else:
        u = rng.random((dates, draws, len(tail)))
        indices = np.argpartition(u, choose - 1, axis=2)[..., :choose]
    mask = np.ones((*indices.shape[:2], len(tail)), dtype=bool)
    np.put_along_axis(mask, indices, False, axis=2)
    retained = np.broadcast_to(tail, mask.shape)[mask].reshape(*mask.shape[:2], len(tail)-choose)
    fixed = np.broadcast_to(fixed, (*mask.shape[:2], len(fixed)))
    return np.concatenate([fixed, retained], axis=2), combinations_n


def residual(model, values, months, idx, target, donors):
    mu, cov = model[:2]
    size = donors.shape[-1]
    unique, match = np.unique(donors.reshape(-1, size), axis=0, return_inverse=True)
    blocks = cov[unique[:, :, None], unique[:, None, :]]
    weights = np.linalg.solve(blocks, cov[target, unique][..., None])[..., 0]
    w = weights[match].reshape(donors.shape)
    anomalies = values[idx[:, None, None], donors] - mu[months[idx, None, None], donors]
    p = mu[months[idx], target, None] + np.sum(w * anomalies, axis=-1)
    return p - values[idx, target, None]


def supported_groups(pools, target, members):
    measured = set(pools)
    for size, groups in members.items():
        for identity, group in enumerate(groups):
            if target in group:
                r = len(measured.intersection(group))
                if r and len(pools) - r >= 3:
                    kept = pools[~np.isin(pools, group)]
                    yield size, identity, r, kept


def evaluate_city(city):
    panel = pd.read_csv(DATA / f'{city}_panel.csv', index_col='date', parse_dates=True).loc['2017':'2025']
    sites = pd.read_csv(DATA / f'{city}_stations.csv', index_col='site').loc[panel.columns]
    km = distances(sites)
    order = np.argsort(km, axis=1, kind='stable')
    full_size = len(sites) * 30 // 100
    groups = {k: library(order, k) for k in sorted(set((*SIZES, full_size)))}
    grp_info = pd.DataFrame([
        (city, size, identity, km[np.ix_(group, group)].max(),
         ' '.join(sites.index[list(group)]))
        for size, known in groups.items() for identity, group in enumerate(known)
    ], columns=['metro', 'size', 'footprint', 'diameter_km', 'members'])
    fits, overlap_info = {}, []
    for label, (start, stop, params) in FITS.items():
        train = panel.loc[start:stop]
        fits[label] = fit(train, km, **params)
        valid = train.notna().to_numpy(np.int64)
        n = (valid.T @ valid)[np.triu_indices(len(sites), 1)]
        limit = params.get('min_overlap', 60)
        overlap_info.append((city, label, start, stop, limit, len(n), (n < limit).sum(), n.min()))

    values = panel.to_numpy()
    years, months = panel.index.year.to_numpy(), panel.index.month.to_numpy()-1
    roster, bucket = np.unique(np.isfinite(values), axis=0, return_inverse=True)
    rng = np.random.default_rng(17731)
    primary, dose, per_fit, split_rows = [], [], [], []
    for k, reported in enumerate(roster):
        idx = np.flatnonzero((bucket == k) & (years >= 2021))
        if reported.sum() < 9 or not len(idx):
            continue
        test = idx[years[idx] >= 2023]
        for i in np.flatnonzero(reported):
            available = order[i][reported[order[i]] & (order[i] != i)]
            designs = list(supported_groups(available, i, groups))
            if not designs:
                continue
            counts = 0
            full_errors = {name: np.zeros((2, len(test))) for name in fits}
            for drop in sorted({item[2] for item in designs}):
                picked = [item for item in designs if item[2] == drop]
                sizes = np.array([item[0] for item in picked])
                ids = np.array([item[1] for item in picked])
                donors = np.stack([item[3] for item in picked])
                main = sizes == full_size
                width = donors.shape
                if main.any():
                    kept = np.broadcast_to(donors[main], (len(idx), main.sum(), width[1]))
                    split = {name: residual(fits[name], values, months, idx, i, kept).ravel()
                             for name in ['early', 'late']}
                    split_rows.append(pd.DataFrame({
                        'date': panel.index[idx].repeat(main.sum()), 'site': sites.index[i], **split
                    }))
                if not len(test):
                    continue

                kept = np.broadcast_to(donors, (len(test), *width))
                fixed = residual(fits['full'], values, months, test, i, kept)
                random, n = sampled_pool(available, drop, len(test), rng)
                rnd = residual(fits['full'], values, months, test, i, random)
                base_mae = np.abs(rnd).mean(axis=1)
                delta = np.abs(fixed) - base_mae[:, None]
                dose.append(pd.DataFrame({
                    'date': panel.index[test].repeat(width[0]), 'site': sites.index[i],
                    'size': np.tile(sizes, len(test)), 'removed': drop, 'pool': len(available),
                    'delta': delta.ravel(),
                }))
                if not main.any():
                    continue

                m_kept = kept[:, main]
                num = main.sum()
                counts += num
                for name, model in fits.items():
                    if name == 'full':
                        g, u = fixed[:, main], rnd
                    else:
                        g = residual(model, values, months, test, i, m_kept)
                        u = residual(model, values, months, test, i, random)
                    full_errors[name][0] += np.abs(g).sum(axis=1)
                    full_errors[name][1] += np.abs(u).mean(axis=1) * num

                far = np.broadcast_to(available[drop:], (len(test), 1, width[1]))
                near_signed = residual(fits['full'], values, months, test, i, far)[:, 0]
                second = rnd if n <= 32 else residual(
                    fits['full'], values, months, test, i, sampled_pool(available, drop, len(test), rng)[0])
                rank = np.argmax(available[None, :] == donors[main, :1], axis=1)
                matched = np.zeros((len(test), num))
                possibilities = np.zeros(num, dtype=int)
                for j in np.unique(rank):
                    option, count = sampled_pool(available, drop, len(test), rng, rank=j)
                    rr = residual(fits['full'], values, months, test, i, option)
                    matched[:, rank == j] = np.abs(rr).mean(axis=1)[:, None]
                    possibilities[rank == j] = count
                primary.append(pd.DataFrame({
                    'date': panel.index[test].repeat(num), 'site': sites.index[i],
                    'footprint': np.tile(ids[main], len(test)), 'actual_y': np.repeat(values[test, i], num),
                    'distance_km': np.tile(km[i, donors[main, 0]], len(test)),
                    'admissible': np.tile(possibilities, len(test)), 'removed': drop, 'pool': len(available),
                    'group_abs': np.abs(fixed[:, main]).ravel(), 'group_signed': fixed[:, main].ravel(),
                    'random_abs': np.repeat(base_mae, num),
                    'random2_abs': np.repeat(np.abs(second).mean(axis=1), num),
                    'random_signed': np.repeat(rnd.mean(axis=1), num),
                    'nearest_signed': np.repeat(near_signed, num), 'rank_abs': matched.ravel(),
                }))
            if counts:
                for name, (g, u) in full_errors.items():
                    per_fit.append(pd.DataFrame(dict(date=panel.index[test], site=sites.index[i],
                                                     fit=name, grouped_mae=g/counts, random_mae=u/counts)))
    main = pd.concat(primary, ignore_index=True)
    expected = pd.read_csv(RESULTS / f'{city}_review_footprint_predictions.csv.gz', parse_dates=['date'])
    expected = expected[expected.date.dt.year.ge(2023)].set_index(['date', 'site', 'footprint'])
    align = main.set_index(['date', 'site', 'footprint']).sort_index()
    expected = expected.sort_index()
    assert align.index.is_unique and align.index.equals(expected.index)
    np.testing.assert_allclose(align.group_signed, expected.empirical - expected.y, atol=1e-10)
    np.testing.assert_allclose(align.distance_km, expected.distance_km, atol=1e-10)
    return (main.assign(metro=city), pd.concat(dose).assign(metro=city),
            pd.concat(per_fit).assign(metro=city), pd.concat(split_rows).assign(metro=city),
            grp_info, overlap_info)


def summarize(city, main, dose, models, split, library_info):
    shape, match, missing, loa, stability, calibration, bias = [], [], [], [], [], [], []
    idx = ['date', 'site']
    d = dose.assign(frac=dose.removed/dose.pool)
    by_size = d.groupby([*idx, 'size'])[['delta', 'frac', 'removed']].mean()
    all_cases = by_size.delta.unstack()
    usable = all_cases.dropna(subset=list(SIZES)).index
    for k, grp in d.groupby('size'):
        a = by_size.xs(k, level='size')
        keep = a.loc[a.index.intersection(usable)]
        diams = library_info.loc[library_info['size'].eq(k), 'diameter_km']
        lo, hi = date_bootstrap(keep.delta.to_numpy(), keep.index.get_level_values('date'), 14)
        shape.append((city, k, len(diams), len(a), len(keep), diams.min(), diams.median(), diams.max(),
                      keep.removed.mean(), keep.frac.mean(), keep.delta.mean(), lo, hi))

    a = main.assign(loss_fraction=main.removed/main.pool,
                    count_limited=main.admissible.eq(1),
                    ordinary=main.group_abs-main.random_abs, rank_matched=main.group_abs-main.rank_abs,
                    simulation=main.random_abs-main.random2_abs)
    for name, cases in [('all', a), ('nonunique', a[a.admissible.gt(1)])]:
        pairs = cases.groupby(idx)[['ordinary','rank_matched','simulation','actual_y','loss_fraction']].mean()
        vector = pairs.rank_matched.to_numpy()
        lo, hi = date_bootstrap(vector, pairs.index.get_level_values('date'), 14)
        match.append((city, name, len(cases), len(pairs), cases.count_limited.mean(),
                      pairs.actual_y.mean(), pairs.loss_fraction.mean(), pairs.ordinary.mean(),
                      vector.mean(), lo, hi))
    pairs = a.groupby(idx).simulation.mean()
    missing.append((city, len(pairs), np.linalg.norm(pairs)/(np.sqrt(2)*len(pairs))))
    for num in sorted(a.footprint.unique()):
        subset = a[a.footprint.ne(num)].groupby(idx).ordinary.mean()
        loa.append((city, num, len(subset), subset.mean()))
    signed = a.groupby(idx)[['group_signed','random_signed','nearest_signed']].mean().reset_index()
    for year, states in signed.groupby(signed.date.dt.year):
        bias.append((city, year, len(states), *states[['group_signed','random_signed','nearest_signed']].mean()))
    for name, row in models.groupby('fit'):
        delta = row.grouped_mae - row.random_mae
        lo, hi = date_bootstrap(delta.to_numpy(), row.date, 14)
        stability.append((city, name, len(row), row.grouped_mae.mean(), row.random_mae.mean(),
                          delta.mean(), lo, hi))
    test = split[split.date.dt.year.ge(2023)]
    cal = split[split.date.dt.year.le(2022)]
    cw = case_weights(cal.assign(y=0))
    tw = case_weights(test.assign(y=0))
    for name in ['early', 'late']:
        qlo, qhi = percentile(cal[name].to_numpy(), cw), percentile(-cal[name].to_numpy(), cw)
        e = test[name].to_numpy()
        covered = np.average((e <= qlo) & (e >= -qhi), weights=tw)
        calibration.append((city, name, round(tw.sum()), covered, qlo+qhi))
    return shape, match, missing, loa, stability, calibration, bias


if __name__ == '__main__':
    libraries, overlap = [], []
    tables = [
        ('group_strength', 'metro size groups available_cases common_cases diameter_min diameter_median diameter_max mean_removed fraction delta lo14 hi14'),
        ('distance_conditioned', 'metro reference mask_rows target_days singleton_fraction observed_mean removed_fraction ordinary_delta conditioned_delta lo14 hi14'),
        ('random_mcse', 'metro target_days mcse'),
        ('leave_group_out', 'metro omitted_group target_days delta'),
        ('covariance_stability', 'metro fit target_days grouped_mae random_mae delta lo14 hi14'),
        ('covariance_split_intervals', 'metro fit target_days coverage width'),
        ('signed_errors', 'metro year target_days grouped random nearest'),
    ]
    records = [[] for _ in tables]
    for metro in REGIONS:
        primary, dose, models, split, lib, counts = evaluate_city(metro)
        primary.to_csv(RESULTS / f'{metro}_referee_geometry_cases.csv.gz', index=False)
        dose.to_csv(RESULTS / f'{metro}_referee_strength_cases.csv.gz', index=False)
        libraries.append(lib)
        overlap.extend(counts)
        for output, rows in zip(records, summarize(metro, primary, dose, models, split, lib)):
            output.extend(rows)
        print(metro, len(primary), len(dose), 'test mask rows', flush=True)
    for ((name, columns), rows) in zip(tables, records):
        pd.DataFrame(rows, columns=columns.split()).to_csv(RESULTS / f'{name}.csv', index=False)
    pd.concat(libraries).to_csv(RESULTS / 'group_libraries_review.csv', index=False)
    cols = ['metro','fit','start','stop','overlap_limit','pairs','zero_filled_pairs','min_common_days']
    pd.DataFrame(overlap, columns=cols).to_csv(RESULTS / 'overlap_review.csv', index=False)
