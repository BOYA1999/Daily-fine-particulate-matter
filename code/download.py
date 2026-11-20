import gzip
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
URL = 'https://aqs.epa.gov/data/api/dailyData/byCBSA'
BOUNDARY_URL = ('https://gis.carb.arb.ca.gov/hosting/rest/services/'
                'California_Air_Resources_Board_Air_Basin_MIL1/MapServer/0/query')
BOUNDARY = RAW / 'south_coast.geojson'
REGIONS = {'LA': '31080', 'NY': '35620', 'CH': '16980'}
QUERY_REGIONS = {**REGIONS, 'IE': '40140'}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        if not BOUNDARY.exists():
            response = session.get(BOUNDARY_URL, params={'where': "AB='SC'", 'outFields': 'NAME,AB',
                                   'outSR': 4326, 'f': 'geojson'}, timeout=40)
            response.raise_for_status()
            features = response.json()['features']
            assert len(features) == 1 and features[0]['properties']['AB'] == 'SC'
            BOUNDARY.write_bytes(response.content)
        for year in range(2017, 2026):
            for city, cbsa in QUERY_REGIONS.items():
                path = RAW / f'{city}_{year}.json.gz'
                if path.exists():
                    continue
                params = dict(email='test@aqs.api', key='test', param='88101',
                              bdate=f'{year}0101', edate=f'{year}1231', cbsa=cbsa)
                for attempt in range(3):
                    start = time.monotonic()
                    try:
                        response = session.get(URL, params=params, timeout=(15, 85))
                        response.raise_for_status()
                        payload = response.json()
                        header = payload['Header'][0]
                        if header['status'] != 'Success' or len(payload['Data']) != header['rows']:
                            raise ValueError(header)
                    except (requests.RequestException, ValueError) as error:
                        if attempt == 2:
                            raise
                        print(city, year, str(error), flush=True)
                        time.sleep(12)
                    else:
                        path.write_bytes(gzip.compress(response.content, mtime=0))
                        print(city, year, header['rows'], header['request_time'], flush=True)
                        time.sleep(max(0, 6.5 - (time.monotonic() - start)))
                        break


if __name__ == '__main__':
    main()
