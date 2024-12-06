import requests
from datetime import date, datetime, timezone, timedelta
import pandas as pd
import json

'''
to retrieve the weather detail from this api

https://api.weather.com/v1/location/WBGW:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate=20241019&endDate=20241019 #maximum 31 days data

https://api.weather.com/v1/location/WBGW:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e
format on url variable

PLEASE CHANGES THIS VARIABLES VALUE:
start_date - start date of the data
end_date - end date of the data
url - change for different place/location
location_name - for naming the csv file based on format {location_name}_{start_date}-{end_date}.csv

'''

'''
mulu: https://api.weather.com/v1/location/WBMU:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e

lawas: https://api.weather.com/v1/location/WBGW:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e

kuching: https://api.weather.com/v1/location/WBGG:9:MY/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e


'''

location = {
    'mulu': 'WBMU:9:MY',
    'lawas': 'WBGW:9:MY',
    'kuching': 'WBGG:9:MY'
}

location_name = 'mulu'

url = f'https://api.weather.com/v1/location/{location[location_name]}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e'
start_date = datetime.strptime('20200101','%Y%m%d')
end_date = datetime.strptime('20220717','%Y%m%d')
print(f"{location_name}/{location_name}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}")

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

# weathers_data.to_csv(f"{location_name}/{location_name}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv", index=False)
weathers_data.to_csv(f"{location_name}/rainfall-feature-wunderground.csv", index=False)