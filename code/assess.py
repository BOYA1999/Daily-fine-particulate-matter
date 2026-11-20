import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from download import REGIONS
from model import FEATURES, POINTS, RESULTS

SCALES = {'constant': [], 'distance': ['distance_km'], 'adaptive': FEATURES}


def point_summary(test):
    rows = []
    for keys, group in test.groupby(['pct', 'pattern']):
        high = group.y.to_numpy() > 35
        for model in POINTS:
            p, y = group[model].to_numpy(), group.y.to_numpy()
            error = p - y
            rows.append((*keys, model, len(y), group.date.nunique(), high.sum(),
                         group.loc[high, 'date'].nunique(), group.donors.mean(),
                         group.distance_km.median(), abs(error).mean(), np.sqrt((error**2).mean()),
                         error[high].mean(), (high & (p <= 35)).sum()))
    return pd.DataFrame(rows, columns=['pct', 'pattern', 'model', 'n', 'dates', 'high', 'high_dates',
                                       'mean_donors', 'median_km', 'mae', 'rmse', 'bias_high', 'miss_high'])


def scaled_intervals(predictions, train, mode):
    result = predictions.copy()
    err_train = train.y - train.empirical
    calibration = result.date.dt.year.le(2022) & result.pct.eq(30) & result.pattern.isin(mode)
    err = result.y - result.empirical
    estimates = []
    for label, features in SCALES.items():
        if features:
            reg = Ridge(alpha=1).fit(np.log1p(train[features].clip(lower=0)),
                                     np.log(abs(err_train).clip(lower=0.25)))
            scale = np.exp(reg.predict(np.log1p(result[features].clip(lower=0))))
        else:
            scale = np.ones(len(result))
        score = (err / scale).loc[calibration]
        qlow = np.quantile(-score, 0.95, method='higher')
        qhigh = np.quantile(score, 0.95, method='higher')
        result[label + '_lo'] = result.empirical - qlow * scale
        result[label + '_hi'] = result.empirical + qhigh * scale
        estimates.append((label, int(calibration.sum()), qlow, qhigh))
    return result, estimates


def subsets(test):
    yield 'all', test
    main = test.loc[test.pct.eq(30) & test.pattern.isin(['random', 'local'])]
    for year, group in main.groupby(main.date.dt.year):
        yield str(year), group
    daily = main.loc[main.pattern.eq('random') & main.y.gt(35)].groupby('date').size()
    busiest = daily.sort_values(ascending=False, kind='stable').index[:5]
    yield 'without_top5', main.loc[~main.date.isin(busiest)]


def interval_summary(test):
    rows = []
    for subset, frame in subsets(test):
        for keys, group in frame.groupby(['pct', 'pattern']):
            y = group.y.to_numpy()
            for name in SCALES:
                lo, hi = group[[name + '_lo', name + '_hi']].to_numpy().T
                width = hi - lo
                coverage = ((lo <= y) & (hi >= y)).mean()
                score = (width + 20 * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))).mean()
                for t in [25, 35]:
                    high, below, flag = y > t, hi <= t, lo > t
                    review = ~(below | flag)
                    rows.append((subset, *keys, name, t, len(y), high.sum(),
                                 group.loc[high, 'date'].nunique(), coverage, (y > hi).mean(),
                                 width.mean(), score, below.sum(), flag.sum(), review.sum(),
                                 (high & below).sum(), (high & flag).sum(), (high & review).sum(),
                                 (~high & flag).sum(), (high & (group.empirical <= t)).sum()))
    return pd.DataFrame(rows, columns=['subset', 'pct', 'pattern', 'scale', 'threshold', 'n', 'high',
                         'high_dates', 'coverage', 'upper_miss', 'width', 'interval_score', 'below',
                         'flags', 'review', 'high_below', 'high_flags', 'high_review', 'false_flags',
                         'point_miss'])


def bootstrap(delta, dates, eligible):
    temp = pd.DataFrame({'delta': np.where(eligible, delta, 0), 'n': eligible,
                         'year': dates.year, 'block': (dates.dayofyear - 1) // 14})
    blocks = temp.groupby(['year', 'block'])[['delta', 'n']].sum()
    rng = np.random.default_rng(1971)
    reps = np.zeros((2000, 2))
    for _, batch in blocks.groupby(level='year'):
        ids = rng.integers(len(batch), size=(len(reps), len(batch)))
        reps += batch.to_numpy()[ids].sum(axis=1)
    valid = reps[:, 1] > 0
    if not eligible.any():
        return np.nan, np.nan, np.nan, 0
    q = np.quantile(reps[valid, 0] / reps[valid, 1], [0.025, 0.975])
    return delta[eligible].mean(), *q, int(eligible.sum())


def paired_summary(test):
    primary = test.loc[test.pct.eq(30)].set_index(['date', 'site'])
    local, random = [primary.loc[primary.pattern.eq(p)].sort_index() for p in ['local', 'random']]
    assert local.index.equals(random.index)
    assert (local.y == random.y).all() and (local.donors == random.donors).all()
    y = random.y.to_numpy()
    dates = random.index.get_level_values('date')
    rows = []
    for name in POINTS:
        p, q = local[name].to_numpy(), random[name].to_numpy()
        rows.append((name, 'local_minus_random_mae',
                     *bootstrap(abs(p - y) - abs(q - y), dates, np.ones(len(y), dtype=bool))))
        for t in [25, 35]:
            rows.append((name, f'local_minus_random_miss_{t}',
                         *bootstrap((p <= t).astype(int) - (q <= t), dates, y > t)))
    for name in ['distance', 'adaptive']:
        p, q = local[name + '_hi'].to_numpy(), local.constant_hi.to_numpy()
        for t in [25, 35]:
            rows.append((name, f'local_minus_constant_false_low_{t}',
                         *bootstrap((p <= t).astype(int) - (q <= t), dates, y > t)))
    return pd.DataFrame(rows, columns=['method', 'contrast', 'estimate', 'lo95', 'hi95', 'n'])


def review_budget(test):
    nearest = test.loc[test.pct.eq(30) & test.pattern.eq('local')].copy()
    rows = []
    for threshold in [25, 35]:
        candidates = nearest.loc[nearest.empirical.le(threshold)].copy()
        point_misses = int(candidates.y.gt(threshold).sum())
        for scale in SCALES:
            candidates['priority'] = ((threshold - candidates.empirical)
                                      / (candidates[scale + '_hi'] - candidates.empirical))
            ordered = candidates.sort_values(['priority', 'site'])
            for budget in [1, 2]:
                reviewed = ordered.groupby('date', sort=False).head(budget)
                captured = reviewed.y.gt(threshold)
                rows.append((scale, threshold, budget, len(reviewed), point_misses, captured.sum(),
                             reviewed.loc[captured, 'date'].nunique()))
    return pd.DataFrame(rows, columns=['scale', 'threshold', 'daily_slots', 'reviewed',
                                       'point_misses', 'high_referred', 'high_dates_referred'])


def main():
    points, intervals, paired, budgets, quantiles = [], [], [], [], []
    for city in REGIONS:
        pred = pd.read_csv(RESULTS / f'{city}_predictions.csv.gz', parse_dates=['date'])
        dev = pd.read_csv(RESULTS / f'{city}_development.csv.gz')
        for mode in [['random'], ['random', 'local']]:
            cal = '+'.join(mode)
            bounds, q = scaled_intervals(pred, dev, mode)
            test = bounds.loc[bounds.date.dt.year.ge(2023)]
            intervals.append(interval_summary(test).assign(metro=city, calibration=cal))
            quantiles.extend((city, cal, *v) for v in q)
            if len(mode) == 1:
                points.append(point_summary(test).assign(metro=city))
                paired.append(paired_summary(test).assign(metro=city))
                budgets.append(review_budget(test).assign(metro=city))
                test.to_csv(RESULTS / f'{city}_test_intervals.csv.gz', index=False)
    for name, parts in [('points', points), ('intervals', intervals), ('paired', paired),
                        ('review_budget', budgets)]:
        pd.concat(parts, ignore_index=True).to_csv(RESULTS / f'{name}.csv', index=False)
    pd.DataFrame(quantiles, columns=['metro', 'calibration', 'scale', 'n', 'q_low', 'q_high']
                 ).to_csv(RESULTS / 'quantiles.csv', index=False)


if __name__ == '__main__':
    main()
