# claude
"""
Scraper for the Caltech ACE ICME catalog:
  https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm

Outputs: icme_catalog.csv in the same directory as this script.

Requirements:
    pip install requests beautifulsoup4
"""

import csv
import re
import requests
from bs4 import BeautifulSoup

URL = "https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"

# Clean column names matching the table headers (in order)
COLUMNS = [
    "disturbance_datetime_ut",
    "icme_plasma_field_start_ut",
    "icme_plasma_field_end_ut",
    "comp_start_hrs",
    "comp_end_hrs",
    "mc_start_hrs",
    "mc_end_hrs",
    "bde",
    "bif",
    "quality",
    "dv_km_s",
    "v_icme_km_s",
    "v_max_km_s",
    "b_nt",
    "mc",
    "dst_nt",
    "v_transit_km_s",
    "lasco_cme_datetime_ut",
]


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def clean_cell(text: str) -> str:
    """Strip whitespace and normalise non-breaking spaces / ellipses."""
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Replace HTML ellipsis placeholder
    if text in ("...", ". . ."):
        return ""
    return text


def is_header_row(row) -> bool:
    """Return True if this <tr> is one of the repeated header rows."""
    return bool(row.find("th") or row.find("b"))


def parse_table(soup: BeautifulSoup) -> list[dict]:
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> found in the page.")

    rows = []
    for tr in table.find_all("tr"):
        if is_header_row(tr):
            continue  # skip repeated header rows

        cells = [clean_cell(td.get_text(" ", strip=True)) for td in tr.find_all("td")]

        # The ICME/plasma start+end are a single <td colspan=2> pair — the
        # HTML sample shows them as two separate <td>s, so we get 18 cells.
        # Guard against stray empty / partial rows.
        if len(cells) < 17:
            continue
        # Pad to 18 if the last column (LASCO CME) is missing
        while len(cells) < 18:
            cells.append("")

        row = dict(zip(COLUMNS, cells[:18]))
        rows.append(row)

    return rows


def main():
    print(f"Fetching {URL} ...")
    html = fetch_html(URL)

    soup = BeautifulSoup(html, "html.parser")
    rows = parse_table(soup)
    print(f"Parsed {len(rows)} data rows.")

    out_file = "data/icme_catalog.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()