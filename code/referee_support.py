import pandas as pd

from download import REGIONS
from model import DATA, RESULTS
from review_geometry import date_bootstrap
from referee_screening import matched_bounds, screening_table


def main():
    smoke_windows = {'NY': ('2023-06-06', '2023-06-08'), 'CH': ('2023-06-26', '2023-06-29')}
    aqs_dates, screens, smoke, eligibility, stations = [], [], [], [], []
    a = pd.read_csv(RESULTS / 'review_geometry_pairs.csv.gz', parse_dates=['date'])
    a = a[a.design.eq('footprint') & a.method.isin(['covariance','fixed_idw'])].copy()
    a['delta'] = a.footprint - a.random
    for metro in REGIONS:
        dates = pd.read_csv(RESULTS / f'{metro}_event_contrast_site_days.csv', parse_dates=['date'])
        dates = dates.loc[dates.date.dt.year.ge(2023)]
        affected = set(dates[dates.changed].date)
        current = a[a.metro.eq(metro) & a.method.eq('covariance')]
        retain = current[~current.date.isin(affected)]
        lo, hi = date_bootstrap(retain.delta.to_numpy(), retain.date, 14)
        aqs_dates.append((metro, len(dates), dates.changed.sum(), len(affected), current.date.nunique()-retain.date.nunique(),
                          len(current), len(retain), retain.delta.mean(), lo, hi))
        for mode, evaluated in matched_bounds(metro).items():
            for row in screening_table(evaluated[~evaluated.date.isin(affected)]):
                screens.append((metro, mode, *row))
            if metro in smoke_windows:
                start, end = smoke_windows[metro]
                in_window = evaluated.date.between(start, end)
                for label, subset in [('inside', evaluated[in_window]), ('outside', evaluated[~in_window])]:
                    smoke.extend((metro, mode, start, end, label, *row) for row in screening_table(subset))
        panel = pd.read_csv(DATA / f'{metro}_panel.csv', index_col='date').loc['2023':]
        exists = panel.notna()
        eligible = exists.loc[exists.sum(axis=1).ge(9)]
        for site in exists:
            count, used = exists[site].sum(), eligible[site].sum()
            reason = 'eligible' if used else 'no valid reading' if not count else 'no nine-station date'
            eligibility.append((metro, site, count, used, reason))
    for (metro, method, site), observations in a.groupby(['metro','method','site']):
        lo, hi = date_bootstrap(observations.delta.to_numpy(), observations.date, 14)
        stations.append((metro, method, site, len(observations), observations.delta.mean(), lo, hi))
    files = [
        ('event_exclusion_points', aqs_dates, 'metro comparative_site_days changed_site_days flagged_dates paired_dates_removed full_cases retained_cases delta lo14 hi14'),
        ('event_exclusion_screening', screens, 'metro geometry threshold scale target_days high high_low high_flag high_review nonhigh_low nonhigh_flag nonhigh_review review_count'),
        ('smoke_window_screening', smoke, 'metro geometry start end subset threshold scale target_days high high_low high_flag high_review nonhigh_low nonhigh_flag nonhigh_review review_count'),
        ('roster_audit', eligibility, 'metro site reported_site_days eligible_site_days status'),
        ('station_pointwise_intervals', stations, 'metro method site target_days delta lo14 hi14'),
    ]
    for name, rows, cols in files:
        pd.DataFrame(rows, columns=cols.split()).to_csv(RESULTS / f'{name}.csv', index=False)


if __name__ == '__main__':
    main()
