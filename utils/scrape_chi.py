import urllib.request
import csv
import os
from bs4 import BeautifulSoup

def download_and_parse():
    url = "http://space.ustc.edu.cn/dreams/wind_icmes/"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('gb2312', errors='ignore')
    except Exception as e:
        print(f"Failed to download data: {e}")
        return
    
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    
    data_table = None
    for t in tables:
        rows = t.find_all('tr')
        if not rows:
            continue
        first_row_cells = [c.text.strip() for c in rows[0].find_all(['th', 'td'])]
        if "Start of the Ejecta" in first_row_cells:
            data_table = t
            break
            
    if not data_table:
        print("Could not find the ICME data table.")
        return
        
    rows = data_table.find_all('tr')
    
    header_cells = [c.text.strip() for c in rows[0].find_all(['th', 'td'])]
    try:
        start_idx = header_cells.index("Start of the Ejecta")
        end_idx = header_cells.index("End of the Ejecta")
    except ValueError:
        print("Could not find required columns in the table.")
        return
        
    os.makedirs('data', exist_ok=True)
    out_file = 'data/chi_icme_catalogue.csv'
    
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['icme_plasma_field_start_ut', 'icme_plasma_field_end_ut'])
        
        for row in rows[2:]:
            cells = [c.text.strip() for c in row.find_all(['th', 'td'])]
            if len(cells) > max(start_idx, end_idx):
                start_str = cells[start_idx].replace('T', ' ')
                end_str = cells[end_idx].replace('T', ' ')
                # Ignore empty or malformed rows (like '------' if it existed in these columns)
                if start_str and end_str and '----' not in start_str:
                    writer.writerow([start_str, end_str])
                    
    print(f"Successfully wrote {out_file}")

if __name__ == '__main__':
    download_and_parse()
