import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from download import REGIONS, ROOT

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
POINTS = ['historical_mean', 'persistence', 'idw', 'anomaly_idw', 'isotropic', 'empirical']
FEATURES = ['distance_km', 'conditional_sd', 'empirical', 'donor_sd']


def distances(positions):
    lat, lon = np.radians(positions[['latitude', 'longitude']].to_numpy()).T
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(hav))


def fit(train, km, *, min_overlap=60, eigen_floor=1e-5, shrink=0.1):
    means = train.groupby(train.index.month).mean().fillna(train.mean()).to_numpy()
    anomalies = train - means[train.index.month - 1]
    raw = anomalies.cov(min_periods=min_overlap).fillna(0).to_numpy()
    sd = np.sqrt(raw.diagonal())
    assert (sd > 0).all()
    corr = raw / np.outer(sd, sd)
    eig, vectors = np.linalg.eigh(corr)
    psd = (vectors * eig.clip(eigen_floor)) @ vectors.T
    norm = np.sqrt(psd.diagonal())
    empirical = ((1 - shrink) * psd / np.outer(norm, norm) + shrink * np.eye(len(sd))) * np.outer(sd, sd)

    a, b = np.triu_indices(len(sd), 1)
    valid = (anomalies.notna().to_numpy(dtype=int).T @ anomalies.notna().to_numpy(dtype=int))[a, b] >= min_overlap
    target_corr, distance = corr[a[valid], b[valid]].clip(-1, 1), km[a[valid], b[valid]]
    isotropic_corr = lambda theta, distance: theta[0] * np.exp(-distance / theta[1])
    params = least_squares(lambda theta: isotropic_corr(theta, distance) - target_corr,
                           [0.8, 45], bounds=([0, 1], [0.99, 2000])).x
    isotropic = (isotropic_corr(params, km) + (1 - params[0]) * np.eye(len(sd))) * np.outer(sd, sd)
    return means, empirical, isotropic, params


def predict(panel, km, fitted, first, last, oof=False):
    means, empirical, isotropic, _ = fitted
    rng = np.random.default_rng(20261120)
    values = panel.to_numpy()

    @lru_cache(maxsize=40000)
    def coefficients(target, donors):
        ids = np.asarray(donors)
        ds = km[target, ids]
        w = 1 / np.maximum(ds, 1) ** 2
        w /= w.sum()
        pair = np.ix_(ids, ids)
        wc = np.linalg.solve(empirical[pair], empirical[target, ids])
        wi = np.linalg.solve(isotropic[pair], isotropic[target, ids])
        s = np.sqrt(max(empirical[target, target] - wc @ empirical[target, ids], 1e-5))
        return w, wc, wi, s, ds.min()

    records = []
    for i in np.flatnonzero((panel.index.year >= first) & (panel.index.year <= last)):
        day = panel.index[i]
        current = values[i]
        mu = means[day.month - 1]
        reporting = np.flatnonzero(np.isfinite(current))
        if len(reporting) < 9:
            continue
        for target in reporting:
            donors = reporting[reporting != target]
            order = donors[np.argsort(km[target, donors], kind='stable')]
            random_order = rng.permutation(donors)
            previous = values[i - 1, target] if i else np.nan
            persistence = previous if np.isfinite(previous) else mu[target]
            masks = [] if oof else [(0, 'isolated', donors)]
            for pct in ([30] if oof else [15, 30, 50]):
                removed = len(donors) * pct // 100
                masks.append((pct, 'random', random_order[removed:]))
                if not oof:
                    masks.append((pct, 'local', order[removed:]))
            if not oof:
                masks.append((30, 'random2', rng.permutation(donors)[len(donors) * 30 // 100:]))
            for pct, pattern, use in masks:
                use = np.sort(use)
                w, wc, wi, sd, distance = coefficients(target, tuple(use))
                x = current[use]
                z = x - mu[use]
                records.append((day, panel.columns[target], pct, pattern, len(use), current[target],
                                mu[target], persistence, w @ x, mu[target] + w @ z,
                                mu[target] + wi @ z, mu[target] + wc @ z, distance, sd, x.std()))
    return pd.DataFrame.from_records(records, columns=['date', 'site', 'pct', 'pattern', 'donors', 'y',
                                                        *POINTS, 'distance_km', 'conditional_sd', 'donor_sd'])


def main(city):
    panel = pd.read_csv(DATA / f'{city}_panel.csv', index_col='date', parse_dates=['date'])
    stations = pd.read_csv(DATA / f'{city}_stations.csv', index_col='site').loc[panel.columns]
    km = distances(stations)
    residual_frames = []
    for year in [2019, 2020]:
        trained = fit(panel.loc[:str(year - 1)], km)
        residual_frames.append(predict(panel, km, trained, year, year, oof=True))
    fitted = fit(panel.loc[:'2020'], km)
    RESULTS.mkdir(exist_ok=True)
    pd.concat(residual_frames).to_csv(RESULTS / f'{city}_development.csv.gz', index=False)
    scored = predict(panel, km, fitted, 2021, 2025)
    scored.to_csv(RESULTS / f'{city}_predictions.csv.gz', index=False)
    print(city, len(panel.columns), len(scored), 'isotropic correlation parameters', fitted[-1], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('city', choices=REGIONS)
    main(parser.parse_args().city)
