import requests
from datetime import date, datetime, timezone, timedelta
import pandas as pd
import json
import os
import argparse

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format. Please use YYYYMMDD format")

def setup_parser():
    parser = argparse.ArgumentParser(description='Retrieve historical weather data for Malaysian locations')
    parser.add_argument('location', choices=[
        'mulu', 'lawas', 'kuching', 'miri', 'sibu',
        'kotakinabalu', 'sandakan', 'subang', 'kuantan',
        'bayanlepas', 'senai', 'kotabharu', 'labuan', 'klia'
    ], help='Location name to retrieve weather data for')
    parser.add_argument('--start-date', type=parse_date, required=True,
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', type=parse_date, required=True,
                        help='End date in YYYYMMDD format')
    parser.add_argument('--output-dir', default=None,
                        help='Optional custom output directory (defaults to location name)')
    return parser

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
            'Pressure': obs['pressure'],
            'Wind Direction': obs['wdir'],
        }
        details.append(observation_data)
    return details

def unix_to_datetime(unix_timestamp, timezone_offset):
    utc_datetime = datetime.fromtimestamp(unix_timestamp, timezone.utc)
    timezone_aware_datetime = utc_datetime.astimezone(timezone(offset=timedelta(hours=timezone_offset)))
    return timezone_aware_datetime

def main():
    location = {
        # SARAWAK
        'mulu': 'WBMU:9:MY', # miri
        'lawas': 'WBGW:9:MY', # labuan
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

    parser = setup_parser()
    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("End date must be after start date")

    output_dir = args.output_dir or args.location
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datapath = f"{output_dir}/rainfall-feature-wunderground.csv"
    url = f'https://api.weather.com/v1/location/{location[args.location]}/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e'

    print(f"\nRetrieving weather data for {args.location}")
    print(f"Output path: {datapath}")

    weathers_data = pd.DataFrame()
    iterate_date = args.start_date

    while iterate_date <= args.end_date:
        next_date = min(iterate_date + timedelta(days=30), args.end_date + timedelta(days=1))
        if next_date > args.end_date:
            next_date = args.end_date
        
        api_url = f"{url}&startDate={iterate_date.strftime('%Y%m%d')}&endDate={next_date.strftime('%Y%m%d')}"
        print(f"Fetching data from {iterate_date.strftime('%Y-%m-%d')} to {next_date.strftime('%Y-%m-%d')}")
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            json_page = response.json()
            weather_observation = weather_details(json_page)
            weather_observation = pd.DataFrame(weather_observation)
            weathers_data = pd.concat([weathers_data, weather_observation], axis=0)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            continue

        iterate_date = next_date + timedelta(days=1)

    weathers_data['Time'] = weathers_data['Time'].apply(lambda x: unix_to_datetime(x, 8))
    weathers_data.to_csv(datapath, index=False)
    print(f"Successfully saved the data into {datapath}")

if __name__ == '__main__':
    main()