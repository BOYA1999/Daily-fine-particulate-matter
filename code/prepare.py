import gzip
import json
from hashlib import sha256

import numpy as np
import pandas as pd
from shapely import intersects_xy
from shapely.geometry import shape

from download import BOUNDARY, QUERY_REGIONS, RAW, REGIONS, ROOT

DATA = ROOT / 'data'


def ingest(city):
    inputs, metadata = [], []
    regions = ['LA', 'IE'] if city == 'LA' else [city]
    for region, year in [(r, y) for r in regions for y in range(2017, 2026)]:
        path = RAW / f'{region}_{year}.json.gz'
        raw = path.read_bytes()
        response = json.loads(gzip.decompress(raw))
        frame = pd.DataFrame(response['Data'])
        assert len(frame) == response['Header'][0]['rows']
        assert set(frame.cbsa_code) == {QUERY_REGIONS[region]}
        assert frame.date_local.str.startswith(str(year)).all()
        assert set(frame.parameter_code) == {'88101'}
        metadata.append((region, year, len(frame), sha256(raw).hexdigest(),
                         response['Header'][0]['request_time']))
        inputs.append(frame)
    raw = pd.concat(inputs, ignore_index=True)
    input_rows = len(raw)
    if city == 'LA':
        feature, = json.loads(BOUNDARY.read_bytes())['features']
        basin = shape(feature['geometry'])
        assert basin.is_valid and feature['properties']['AB'] == 'SC'
        raw = raw.loc[intersects_xy(basin, raw.longitude.to_numpy(), raw.latitude.to_numpy())]
    clean = raw.loc[
        raw.sample_duration.isin(['24 HOUR', '24-HR BLK AVG'])
        & raw.pollutant_standard.eq('PM25 24-hour 2006')
        & raw.event_type.isin(['No Events', 'Events Included'])
        & raw.validity_indicator.eq('Y')
        & raw.observation_percent.ge(75)
        & raw.arithmetic_mean.notna()
    ].copy()
    assert set(clean.units_of_measure) == {'Micrograms/cubic meter (LC)'}
    clean['site'] = clean.state_code + '-' + clean.county_code + '-' + clean.site_number
    clean['date'] = pd.to_datetime(clean.date_local)
    clean['year'] = clean.date.dt.year
    instrument = clean.groupby(['site', 'date', 'poc']).arithmetic_mean
    assert instrument.nunique().max() == 1, 'Conflicting instrument-day summaries'
    site_day = instrument.first().groupby(['site', 'date']).median().rename('pm25')
    historical = site_day.reset_index().query('date < "2021-01-01"')
    counts = historical.groupby(['site', historical.date.dt.year]).size().unstack()
    keep = counts.index[(counts >= 90).sum(axis=1) >= 3]
    dating = pd.date_range('2017-01-01', '2025-12-31')
    panel = site_day.unstack('site').reindex(index=dating, columns=keep)
    panel.index.name = 'date'
    assert panel.columns.is_unique
    coords = clean.loc[clean.site.isin(keep) & clean.year.le(2020)]
    grouped = coords.groupby('site')
    stations = grouped[['latitude', 'longitude']].median().reindex(keep)
    stations['name'] = grouped.local_site_name.first()
    position_range = grouped[['latitude', 'longitude']].agg(np.ptp).max()
    print(city, 'raw rows', len(raw), 'valid rows', len(clean), 'sites', len(keep),
          'coordinate span (deg)', position_range.to_dict(), flush=True)
    observed = panel.notna().sum(axis=1)
    print(pd.DataFrame({'observed': observed, 'paired': observed.where(observed >= 9, 0)})
          .groupby(dating.year).sum().to_string(), flush=True)
    panel.to_csv(DATA / f'{city}_panel.csv')
    stations.to_csv(DATA / f'{city}_stations.csv')
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    # Paired summaries isolate days with a change in the event-excluded version.
    keys = ['state_code', 'county_code', 'site_number', 'date_local', 'poc', 'sample_duration', 'pollutant_standard']
    fields = ['observation_count', 'arithmetic_mean', 'validity_indicator']
    combined = clean.loc[clean.site.isin(keep) & clean.event_type.eq('Events Included'),
                         [*keys, *fields, 'date', 'site']]
    excluding = raw.loc[raw.event_type.isin(['Events Excluded', 'Concurred Events Excluded']), [*keys, *fields]]
    paired = combined.merge(excluding, on=keys, suffixes=('_in', '_out'))
    changed = (paired.observation_count_in.ne(paired.observation_count_out)
               | paired.validity_indicator_in.ne(paired.validity_indicator_out)
               | ~np.isclose(paired.arithmetic_mean_in, paired.arithmetic_mean_out, equal_nan=True))
    paired.assign(changed=changed).groupby(['date','site']).changed.any().to_csv(
        output / f'{city}_event_contrast_site_days.csv')
    flow = dict(metro=city, api_rows=input_rows, regional_rows=len(raw), screened_rows=len(clean),
                instrument_days=instrument.ngroups, all_site_days=len(site_day),
                screened_sites=site_day.index.get_level_values('site').nunique(),
                roster_sites=len(keep))
    return metadata, flow


if __name__ == '__main__':
    batches, flows = zip(*(ingest(city) for city in REGIONS))
    records = [record for batch in batches for record in batch]
    pd.DataFrame(records, columns=['query_region', 'year', 'api_rows', 'sha256_gzip', 'request_time']
                 ).to_csv(DATA / 'sources.csv', index=False)
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    pd.DataFrame(flows).to_csv(output / 'input_flow.csv', index=False)
