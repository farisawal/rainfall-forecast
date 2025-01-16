import requests
from datetime import date, datetime, timezone, timedelta
import pandas as pd
import json
import os

'''
to retrieve the weather detail from this api

https://api.weather.com/v1/location/WBGW:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20241019&endDate=20241019 #maximum 31 days data

https://api.weather.com/v1/location/WBGW:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e
format on url variable

PLEASE CHANGES THIS VARIABLES VALUE:
start_date - start date of the data
end_date - end date of the data
location_name - the location of intended data located, but please define its location id in 'location' dictionary

'''

'''
mulu: https://api.weather.com/v1/location/WBMU:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e

lawas: https://api.weather.com/v1/location/WBGW:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e

kuching: https://api.weather.com/v1/location/WBGG:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e

subang: https://api.weather.com/v1/location/WMSA:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e

'''

location = {
    'mulu': 'WBMU:9:MY', # miri
    'lawas': 'WBGW:9:MY', # labuan

    # SARAWAK
    'kuching': 'WBGG:9:MY',
    'miri': 'WBGR:9:MY',
    'sibu': 'WBGS:9:MY',
    # SABAH
    'kotakinabalu': 'WBKK:9:MY',
    'sandakan': 'WBKS:9:MY',
    # SEMENANJUNG
    'subang': 'WMSA:9:MY',
    'kuantan': 'WMKD:9:MY',
    'bayanlepas': 'WMKP:9:MY',
    'senai': 'WMKJ:9:MY',
    'kotabharu': 'WMKC:9:MY',
    # WILAYAH
    'labuan': 'WBKL:9:MY',
    'klia': 'WMKK:9:MY',
    
}

location_name = 'mulu'
start_date = datetime.strptime('20101125','%Y%m%d')
end_date = datetime.strptime('20220717','%Y%m%d')

if not os.path.exists(location_name):
    os.makedirs(location_name)

datapath = f"{location_name}/rainfall-feature-wunderground.csv"

url = f'https://api.weather.com/v1/location/{location[location_name]}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e'

print(f"\n{datapath}")

def weather_details(weather_dict):
    observations = weather_dict['observations']
    details = []
    for obs in observations:
        observation_data = {
            'Time': obs['valid_time_gmt'],
            'Temperature': obs['temp'],
            'Dew Point': obs['dewPt'],
            'Humidity': obs['rh'],
            'Visibility': obs['vis'],
            'Wind Speed': obs['wspd'],
            'Wind Gust': obs['gust'],
            'Pressure': obs['pressure'],
            'Precip.': obs['precip_total'],
            'Wind': obs['wdir_cardinal'],
            'Condition': obs['wx_phrase'],
        }
        details.append(observation_data)
    return details

def unix_to_datetime(unix_timestamp, timezone_offset):
    utc_datetime = datetime.fromtimestamp(unix_timestamp, timezone.utc)
    timezone_aware_datetime = utc_datetime.astimezone(timezone(offset=timedelta(hours=timezone_offset)))
    return timezone_aware_datetime

weathers_data = pd.DataFrame()
iterate_date = start_date

# while start_date < datetime.now():
while iterate_date <= end_date:
    next_date = min(iterate_date + timedelta(days=30), end_date + timedelta(days=1))
    if next_date>end_date:
        next_date = end_date
    api_url = f"{url}&startDate={iterate_date.strftime('%Y%m%d')}&endDate={next_date.strftime('%Y%m%d')}"
    print(api_url)
    page = requests.get(api_url).text
    json_page = json.loads(page)
    weather_observation = weather_details(json_page)
    weather_observation = pd.DataFrame(weather_observation)
    weathers_data = pd.concat([weathers_data, weather_observation], axis=0)
    # start_date += timedelta(days=1)
    iterate_date = next_date + timedelta(days=1)

weathers_data['Time'] = weathers_data['Time'].apply(lambda x: unix_to_datetime(x, 8))

# # weathers_data.to_csv(f"{location_name}/{location_name}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv", index=False)

weathers_data.to_csv(datapath, index=False)
print(f"Successfully saved the data into {datapath}")