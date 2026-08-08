import os
import csv
import urllib.request

URL = "https://stereo-ssc.nascom.nasa.gov/data/ins_data/impact/level3/LanJian_STEREO_ICME_List.txt"
OUTPUT_FILE = "data/jian_icme_catalogue.csv"

def parse_time(t_str):
    t_str = t_str.strip('Z')
    if '.' in t_str:
        t_str = t_str.split('.')[0]
    t_str = t_str.replace('T', ' ')
    return t_str

def main():
    print(f"Downloading data from {URL}...")
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        content = response.read().decode('utf-8')
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    rows = []
    
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        parts = line.split('\t')
        if len(parts) < 2:
            # Fallback for spacing
            parts = line.split()
            
        if len(parts) >= 2:
            start_str = parts[0]
            end_str = parts[1]
            
            start_time = parse_time(start_str)
            end_time = parse_time(end_str)
            
            rows.append({
                'icme_plasma_field_start_ut': start_time,
                'icme_plasma_field_end_ut': end_time
            })
            
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['icme_plasma_field_start_ut', 'icme_plasma_field_end_ut']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully processed {len(rows)} records and saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
