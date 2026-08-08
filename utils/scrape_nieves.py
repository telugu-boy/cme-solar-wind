import os
import json
import requests
import pandas as pd

def scrape_nieves():
    url = "https://wind.nasa.gov/ICME_catalog/ICMEcatalog_Wind_web.json"
    
    print(f"Downloading data from {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    json_data = response.json()
    data = json_data['data']
    
    df = pd.DataFrame(data)
    
    # Rename columns to match requested names
    rename_mapping = {
        'ICME_datestart': 'icme_plasma_field_start_ut',
        'ICME_dateend': 'icme_plasma_field_end_ut'
    }
    df.rename(columns=rename_mapping, inplace=True)
    
    # Format datetime columns to YYYY-MM-DD HH:MM:SS
    for col in ['icme_plasma_field_start_ut', 'icme_plasma_field_end_ut']:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'nieves_icme_catalogue.csv')
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df)} records to {output_file}")

if __name__ == "__main__":
    scrape_nieves()
