import numpy as np
import pandas as pd

from model import RESULTS
from review_geometry import date_bootstrap
from review_intervals import SCALES, case_weights, percentile, scales, summarize


TRAIN_REGIONS = ('LA', 'NY')
EVAL_REGION = 'CH'


def development(city):
    return pd.read_csv(RESULTS / f'{city}_development.csv.gz').assign(metro=city)


def footprints(city):
    return pd.read_csv(RESULTS / f'{city}_review_footprint_predictions.csv.gz',
                       parse_dates=['date']).assign(metro=city)


def fit_bounds(train, calibration, test, train_weights=None, calibration_weights=None):
    past_scale = scales(train, calibration, train_weights)
    test_scale = scales(train, test, train_weights)
    future = test.copy()
    for name in SCALES:
        score = (calibration.y - calibration.empirical).to_numpy() / past_scale[name]
        future[name + '_lo'] = test.empirical - test_scale[name] * percentile(-score, calibration_weights)
        future[name + '_hi'] = test.empirical + test_scale[name] * percentile(score, calibration_weights)
    return future


def combined_metrics(frame):
    y, lo, hi = frame[['y', 'combined_lo', 'combined_hi']].to_numpy().T
    return {'coverage': ((lo <= y) & (y <= hi)).astype(int),
            'score': hi - lo + 20 * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))}


def main():
    train = pd.concat([development(c) for c in TRAIN_REGIONS], ignore_index=True)
    train_sizes = train.groupby('metro').y.transform('size')
    train_weights = len(train) / (len(TRAIN_REGIONS) * train_sizes)
    source = pd.concat([footprints(c).loc[lambda f: f.date.dt.year.le(2022)]
                        for c in TRAIN_REGIONS], ignore_index=True)
    source_weights = source.assign(w=case_weights(source))
    source_weights = (source_weights.w / source_weights.groupby('metro').w.transform('sum')).to_numpy()

    target = footprints(EVAL_REGION)
    local = target[target.date.dt.year.le(2022)]
    test = target[target.date.dt.year.ge(2023)]
    weights = case_weights(local)
    regimes = {
        'source_only': fit_bounds(train, source, test, train_weights, source_weights),
        'target_recalibrated': fit_bounds(train, local, test, train_weights, weights),
        'target_only': fit_bounds(development(EVAL_REGION), local, test, calibration_weights=weights),
    }
    summary, comparisons = [], []
    raw = combined_metrics(regimes['source_only'])
    for name, frame in regimes.items():
        results, _ = summarize(frame)
        summary.extend((name, *r) for r in results)
        if name == 'source_only':
            continue
        for metric, value in combined_metrics(frame).items():
            delta = frame.assign(value=value - raw[metric]).groupby(['date', 'site']).value.mean()
            low, high = date_bootstrap(delta.to_numpy(), delta.index.get_level_values('date'), 14)
            comparisons.append((name, metric, delta.mean(), low, high))
    columns = ['regime', 'scale', 'threshold', 'scenario_rows', 'site_days', 'high', 'high_dates', 'coverage',
               'upper_miss', 'width', 'interval_score', 'reviewed', 'high_below', 'high_flagged',
               'high_referred', 'false_flagged']
    pd.DataFrame(summary, columns=columns).to_csv(RESULTS / 'CH_transfer_intervals.csv', index=False)
    cols = ['regime', 'metric', 'change_from_source_only', 'lo14', 'hi14']
    pd.DataFrame(comparisons, columns=cols).to_csv(RESULTS / 'CH_transfer_paired.csv', index=False)


if __name__ == '__main__':
    main()
