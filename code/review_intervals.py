import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from download import REGIONS
from model import RESULTS
from review_geometry import date_bootstrap

SCALES = {
    'constant': [],
    'distance': ['distance_km'],
    'geometry': ['distance_km', 'conditional_sd'],
    'concentration': ['empirical', 'donor_sd'],
    'combined': ['distance_km', 'conditional_sd', 'empirical', 'donor_sd'],
}


def subset(pred, design, start, end):
    return pred[design].loc[pred[design].date.dt.year.between(start, end)].copy()


def case_weights(cases):
    return 1 / cases.groupby(['date', 'site']).y.transform('size').to_numpy()


def percentile(x, w):
    if np.ptp(w) == 0:
        return np.quantile(x, 0.95, method='higher')
    order = np.argsort(x)
    return x[order][np.searchsorted(np.cumsum(w[order]), 0.95 * w.sum())]


def scales(train, cases, sample_weight=None):
    err = np.log((train.y - train.empirical).abs().clip(lower=0.25))
    outputs = {}
    for name, features in SCALES.items():
        if features:
            model = Ridge(alpha=1).fit(np.log1p(train[features].clip(lower=0)), err,
                                       sample_weight=sample_weight)
            outputs[name] = np.exp(model.predict(np.log1p(cases[features].clip(lower=0))))
        else:
            outputs[name] = np.ones(len(cases))
    return outputs


def predict_bounds(pred, scaled, cal, design):
    past = subset(pred, cal, 2021, 2022)
    future = subset(pred, design, 2023, 2025)
    residual = (past.y - past.empirical).to_numpy()
    w = case_weights(past)
    for name in SCALES:
        r = residual / scaled[cal][name][pred[cal].date.dt.year.le(2022).to_numpy()]
        s = scaled[design][name][pred[design].date.dt.year.ge(2023).to_numpy()]
        future[name + '_lo'] = future.empirical - s * percentile(-r, w)
        future[name + '_hi'] = future.empirical + s * percentile(r, w)
    return future


def summarize(cases):
    rows, contrasts = [], []
    weights = case_weights(cases)
    avg = lambda x: np.average(x, weights=weights)
    y = cases.y.to_numpy()
    metrics = {}
    for name in SCALES:
        lo, hi = cases[[name + '_lo', name + '_hi']].to_numpy().T
        width = hi - lo
        score = width + 20 * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        coverage = (lo <= y) & (y <= hi)
        metrics[name] = {'score': score, 'coverage': coverage.astype(int)}
        references = ['constant', 'concentration'] if name == 'combined' else ['constant']
        for reference in references:
            for metric, values in metrics[name].items():
                delta = values - metrics[reference][metric]
                daily = cases.assign(delta=delta).groupby(['date', 'site']).delta.mean()
                lo95, hi95 = date_bootstrap(daily.to_numpy(), daily.index.get_level_values('date'), 14)
                contrasts.append((name, reference, metric, daily.mean(), lo95, hi95))
        for t in [25, 35]:
            high, below, flagged = y > t, hi <= t, lo > t
            referral = ~(below | flagged)
            rows.append((name, t, len(cases), weights.sum(), (high * weights).sum(),
                         cases.loc[high, 'date'].nunique(), avg(coverage), avg(y > hi),
                         avg(width), avg(score), (referral * weights).sum(),
                         (below * high * weights).sum(), (flagged * high * weights).sum(),
                         (referral * high * weights).sum(), ((~high) * flagged * weights).sum()))
    return rows, contrasts


def episodes(cases):
    high = cases[cases.y.gt(35)].copy()
    days = pd.Series(high.date.unique()).sort_values(ignore_index=True)
    periods = days.diff().dt.days.gt(2).cumsum()
    high['event'] = high.date.map(dict(zip(days, periods)))
    rows = []
    for _, part in high.groupby('event'):
        rows.append((part.date.min(), part.date.max(), part.date.nunique(), len(part),
                     part.empirical.le(35).sum(), part.constant_hi.le(35).sum(),
                     part.combined_hi.le(35).sum()))
    frame = pd.DataFrame(rows, columns=['start', 'end', 'high_dates', 'high_site_days', 'point_miss',
                                        'constant_miss', 'combined_miss'])
    remain = frame.high_site_days.sum() - frame.high_site_days
    different = frame.combined_miss - frame.constant_miss
    frame['leave_out_delta'] = (different.sum() - different) / remain
    return frame


def budget(cases):
    rows = []
    for t in [25, 35]:
        candidate = cases[cases.empirical.le(t)].copy()
        missed = candidate.y.gt(t).sum()
        for name in SCALES:
            candidate['margin'] = (t - candidate.empirical) / (candidate[name + '_hi'] - candidate.empirical)
            ordered = candidate.sort_values(['margin', 'date', 'site'])
            for fraction in [0.0025, 0.005, 0.01, 0.02]:
                queue = ordered.head(int(len(ordered) * fraction))
                reviewed = queue[queue.y.gt(t)]
                rows.append((name, t, fraction, len(candidate), len(queue), missed, len(reviewed),
                             reviewed.date.nunique()))
    return pd.DataFrame(rows, columns=['scale', 'threshold', 'budget_fraction', 'candidates', 'reviewed',
                                       'point_miss', 'confirmed_high', 'high_dates'])


def main():
    results, pairs, all_events, all_queues = [], [], [], []
    for city in REGIONS:
        point = pd.read_csv(RESULTS / f'{city}_predictions.csv.gz', parse_dates=['date']).query('pct == 30')
        dev = pd.read_csv(RESULTS / f'{city}_development.csv.gz')
        footprint = pd.read_csv(RESULTS / f'{city}_review_footprint_predictions.csv.gz', parse_dates=['date'])
        contrasts = [('random', 'local'), ('local', 'local'), ('random', 'footprint'),
                     ('footprint', 'footprint'), ('random', 'random')]
        for center in ['covariance', 'idw']:
            pred = {mode: point[point.pattern.eq(mode)].copy() for mode in ['random', 'local']}
            pred['footprint'] = footprint.copy()
            train = dev.copy()
            if center == 'idw':
                train['empirical'] = train.idw
                for mode, df in pred.items():
                    df['empirical'] = df['fixed_idw' if mode == 'footprint' else 'idw']
            scaled = {mode: scales(train, frame) for mode, frame in pred.items()}
            comparisons = contrasts if center == 'covariance' else [('local', 'local')]
            for cal, seen in comparisons:
                cases = predict_bounds(pred, scaled, cal, seen)
                rows, diffs = summarize(cases)
                results.extend((city, center, cal, seen, *r) for r in rows)
                pairs.extend((city, center, cal, seen, *r) for r in diffs)
                if (center, cal, seen) == ('covariance', 'local', 'local'):
                    all_events.append(episodes(cases).assign(metro=city))
                    all_queues.append(budget(cases).assign(metro=city))
    keys = ['metro', 'center', 'calibration', 'test_design']
    cols = ['scale', 'threshold', 'scenario_rows', 'site_days', 'high', 'high_dates', 'coverage',
            'upper_miss', 'width', 'interval_score', 'reviewed', 'high_below', 'high_flagged',
            'high_referred', 'false_flagged']
    pd.DataFrame(results, columns=keys + cols).to_csv(RESULTS / 'review_interval_summary.csv', index=False)
    cols = ['scale', 'reference', 'metric', 'difference', 'lo14', 'hi14']
    pd.DataFrame(pairs, columns=keys + cols).to_csv(RESULTS / 'review_interval_paired.csv', index=False)
    pd.concat(all_events).to_csv(RESULTS / 'review_episodes.csv', index=False)
    pd.concat(all_queues).to_csv(RESULTS / 'review_queue.csv', index=False)


if __name__ == '__main__':
    main()
