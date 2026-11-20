import numpy as np
import pandas as pd

from download import REGIONS
from model import RESULTS
from review_intervals import case_weights, predict_bounds, scales
from transfer_validation import development, fit_bounds, footprints

SCALES = ('constant', 'combined')
CUTOFFS = (25, 35)
BUDGETS = (.0025, .005, .01, .02)
N_SCHEDULES = 64


def matched_bounds(city):
    mask_rows = footprints(city)
    near = pd.read_csv(RESULTS / f'{city}_predictions.csv.gz', parse_dates=['date'])
    near = near[(near.pct == 30) & near.pattern.eq('local')]
    data = {'footprint': mask_rows, 'local': near}
    scaled = {key: scales(development(city), value) for key, value in data.items()}
    return {mode: predict_bounds(data, scaled, mode, mode) for mode in data}


def screening_table(cases):
    w = case_weights(cases)
    y = cases.y.to_numpy()
    rows = []
    for t in CUTOFFS:
        high = y > t
        for name in SCALES:
            below = cases[name + '_hi'].to_numpy() <= t
            flag = cases[name + '_lo'].to_numpy() > t
            review = ~(below | flag)
            assert not (below & flag).any()
            rows.append((t, name, w.sum(), w[high].sum(),
                         w[high & below].sum(), w[high & flag].sum(), w[high & review].sum(),
                         w[~high & below].sum(), w[~high & flag].sum(), w[~high & review].sum(),
                         w[review].sum()))
    return rows


def budget_rows(cases, schedule):
    rows = []
    for t in CUTOFFS:
        candidates = cases[cases.empirical.le(t)].copy()
        point_high = cases[cases.empirical.gt(t)]
        false_flags = point_high.y.le(t).sum()
        missed = candidates.y.gt(t).sum()
        for name in SCALES:
            allowance = candidates[name + '_hi'] - candidates.empirical
            assert allowance.gt(0).all()
            margin = (t - candidates.empirical) / allowance
            ordered = candidates.assign(priority=margin).sort_values(['priority', 'date', 'site'])
            for budget in BUDGETS:
                reviewed = ordered.head(int(budget*len(candidates)))
                recovered = reviewed.y.gt(t).sum()
                rows.append((schedule, t, budget, name, len(cases), len(candidates), missed,
                             len(reviewed), recovered, len(reviewed)-recovered, false_flags))
    return rows


def schedules(cases, groups):
    days = pd.date_range('2023-01-01', '2025-12-31')
    slot = days.get_indexer(cases.date)
    mask = cases.footprint.to_numpy()
    rng = np.random.default_rng(6008)
    for index in range(N_SCHEDULES):
        chosen = rng.choice(groups, len(days), replace=True)
        selected = cases[mask == chosen[slot]]
        assert not selected.duplicated(['date', 'site']).any()
        yield index, selected


def interval_components(cases, scale):
    w = case_weights(cases)
    lo, hi = cases[[scale + '_lo', scale + '_hi']].to_numpy().T
    y = cases.y.to_numpy()
    rows = [hi-lo, 20*np.maximum(lo-y, 0), 20*np.maximum(y-hi, 0), lo>y, y>hi]
    return tuple(np.average(a, weights=w) for a in rows)


def transfer_sources():
    rows = []
    all_target = footprints('CH')
    local = all_target[all_target.date.dt.year.le(2022)]
    test = all_target[all_target.date.dt.year.ge(2023)]
    for source in ('LA', 'NY'):
        earlier = footprints(source).loc[lambda f: f.date.dt.year.le(2022)]
        for label, cal in [('source', earlier), ('Chicago', local)]:
            cases = fit_bounds(development(source), cal, test, calibration_weights=case_weights(cal))
            width, lower, upper, misslow, misshigh = interval_components(cases, 'combined')
            rows.append((source, label, len(test), width, 1-misslow-misshigh,
                         misslow, misshigh, width+lower+upper))
    cols = ['scale_source', 'quantile_source', 'mask_rows', 'width', 'coverage',
            'lower_miss', 'upper_miss', 'score']
    pd.DataFrame(rows, columns=cols).to_csv(RESULTS / 'transfer_single_source.csv', index=False)


def main():
    decisions, queues, components = [], [], []
    for city in REGIONS:
        bounds = matched_bounds(city)
        masks = pd.read_csv(RESULTS / f'{city}_review_footprints.csv').footprint.unique()
        for geometry, cases in bounds.items():
            decisions.extend((city, geometry, *row) for row in screening_table(cases))
        for schedule, selected in schedules(bounds['footprint'], masks):
            queues.extend((city, 'footprint', *row) for row in budget_rows(selected, schedule))
        queues.extend((city, 'local', *row) for row in budget_rows(bounds['local'], 0))
        for name in SCALES:
            components.append((city, name, *interval_components(bounds['footprint'], name)))
        print(city, len(bounds['footprint']), 'group masks audited', flush=True)
    heads = ['metro','geometry','threshold','scale','target_days','high','high_low','high_flag',
             'high_review','nonhigh_low','nonhigh_flag','nonhigh_review','review_count']
    out = pd.DataFrame(decisions, columns=heads)
    out.to_csv(RESULTS / 'screening_confusion.csv', index=False)
    heads = ['metro','geometry','schedule','threshold','fraction','scale','target_days','candidates',
             'point_miss','reviewed','recovered','reviews_nonhigh','point_false_flags']
    q = pd.DataFrame(queues, columns=heads)
    q.to_csv(RESULTS / 'screening_schedules.csv', index=False)
    keys = ['metro','geometry','threshold','fraction']
    q.drop(columns='schedule').groupby(keys+['scale']).mean().to_csv(RESULTS / 'screening_budget_summary.csv')
    paired = q.pivot(index=[*keys,'schedule','reviewed','point_false_flags'], columns='scale', values='recovered')
    gains = paired.combined-paired.constant
    assert gains.notna().all()
    gains.groupby(level=keys).agg(['size','mean','min','max']).to_csv(RESULTS / 'screening_gains.csv')
    cols = ['metro','scale','width','lower_penalty','upper_penalty','lower_miss','upper_miss']
    pd.DataFrame(components, columns=cols).to_csv(RESULTS / 'interval_score_components.csv', index=False)
    transfer_sources()


if __name__ == '__main__':
    main()
