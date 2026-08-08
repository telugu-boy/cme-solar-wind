import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import math
import os

def parse_decimal_hour(hour_float):
    hours = int(math.floor(hour_float))
    minutes = int(round((hour_float - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return timedelta(hours=hours, minutes=minutes)

def get_datetime_from_doy(year, doy_str, hour_str):
    doy = int(doy_str)
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    dt += parse_decimal_hour(float(hour_str))
    return dt

def scrape_lepping():
    url = 'https://wind.nasa.gov/mfi/mag_cloud_pub1.html'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        html = response.read().decode('latin1')
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        print("No tables found.")
        return
    
    table = tables[0]
    rows = table.find_all('tr')
    
    data = []
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        cell_texts = [c.get_text(strip=True) for c in cells]
        
        # Skip header rows or malformed rows
        if len(cell_texts) < 11:
            continue
        
        try:
            # Check if first cell is a valid Code No.
            code_no = int(cell_texts[0])
        except ValueError:
            continue
            
        y = int(cell_texts[1])
        start_year = 1900 + y if y >= 50 else 2000 + y
        
        start_doy = cell_texts[4]
        start_hour = cell_texts[5]
        
        end_doy = cell_texts[8]
        end_hour = cell_texts[9]
        
        try:
            start_time = get_datetime_from_doy(start_year, start_doy, start_hour)
            
            end_year = start_year
            if int(end_doy) < int(start_doy) and int(start_doy) > 300 and int(end_doy) < 100:
                end_year += 1
                
            end_time = get_datetime_from_doy(end_year, end_doy, end_hour)
            
            data.append({
                'code_no': code_no,
                'icme_plasma_field_start_ut': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'icme_plasma_field_end_ut': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'quality': cell_texts[10]
            })
        except Exception as e:
            print(f"Error parsing row {cell_texts}: {e}")
            continue

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/lepping_icme_catalogue.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} ICMEs to {output_path}")

if __name__ == '__main__':
    scrape_lepping()
