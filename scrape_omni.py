# CLAUDE

"""
NASA OMNI HRO CDF Parallel Downloader
======================================
Replicates:
  wget -r -np -nH --cut-dirs=4 -A cdf \
      https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/
  wget -r -np -nH --cut-dirs=4 -A cdf \
      https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_5min/

Output layout mirrors the remote year/ structure:
  data/omni_hro2_1min/<year>/<file>.cdf
  data/omni_hro2_5min/<year>/<file>.cdf
"""

import os
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

DATASETS = {
    "omni_hro2_1min": "https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/",
    "omni_hro2_5min": "https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_5min/",
}

BASE_OUTPUT_DIR = Path("data")
MAX_WORKERS     = 8          # parallel download threads
CHUNK_SIZE      = 1024 * 256 # 256 KB streaming chunks
RETRIES         = 3          # per-file retry attempts
RETRY_DELAY     = 5          # seconds between retries
REQUEST_TIMEOUT = 60         # seconds
SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; omni-downloader/1.0)"
}

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTML scraping ──────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    return s


def list_links(session: requests.Session, url: str, suffix: str = "/") -> list[str]:
    """Return all href links from an Apache-style directory listing that end with `suffix`."""
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip parent/sorting links
        if href.startswith("?") or href.startswith("/") or href == "../":
            continue
        if href.endswith(suffix):
            links.append(urljoin(url, href))
    return links


def list_cdf_files(session: requests.Session, year_url: str) -> list[tuple[str, str]]:
    """Return (file_url, filename) pairs for every .cdf file in a year directory."""
    resp = session.get(year_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".cdf"):
            file_url = urljoin(year_url, href)
            files.append((file_url, href))
    return files


# ── Download logic ─────────────────────────────────────────────────────────────

def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    pbar: tqdm,
) -> tuple[bool, str]:
    """
    Stream-download `url` → `dest`.
    Skips if dest already exists with the correct size.
    Returns (success, message).
    """
    # HEAD check for existing file
    if dest.exists():
        try:
            head = session.head(url, timeout=REQUEST_TIMEOUT)
            remote_size = int(head.headers.get("Content-Length", -1))
            if remote_size > 0 and dest.stat().st_size == remote_size:
                pbar.update(1)
                return True, f"SKIP (exists) {dest.name}"
        except Exception:
            pass  # fall through and re-download

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
            tmp.rename(dest)
            pbar.update(1)
            return True, f"OK {dest.name}"
        except Exception as exc:
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                if tmp.exists():
                    tmp.unlink()
                pbar.update(1)
                return False, f"FAIL {dest.name}: {exc}"


# ── Orchestration ──────────────────────────────────────────────────────────────

def collect_tasks(
    session: requests.Session,
    base_url: str,
    output_dir: Path,
) -> list[tuple[str, Path]]:
    """
    Scrape year directories then CDF file listings.
    Returns list of (file_url, local_path) tuples.
    """
    log.info("Scanning year directories at %s", base_url)
    year_urls = list_links(session, base_url, suffix="/")
    log.info("  Found %d year directories", len(year_urls))

    tasks = []
    for year_url in year_urls:
        year = year_url.rstrip("/").split("/")[-1]
        try:
            cdf_files = list_cdf_files(session, year_url)
        except Exception as exc:
            log.warning("  Could not list %s: %s", year_url, exc)
            continue
        for file_url, filename in cdf_files:
            dest = output_dir / year / filename
            tasks.append((file_url, dest))
        log.info("  %s  →  %d files", year, len(cdf_files))

    return tasks


def download_dataset(
    dataset_name: str,
    base_url: str,
    output_dir: Path,
    workers: int,
) -> None:
    session = make_session()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Dataset : %s", dataset_name)
    log.info("Remote  : %s", base_url)
    log.info("Local   : %s", output_dir)
    log.info("=" * 60)

    tasks = collect_tasks(session, base_url, output_dir)
    total = len(tasks)
    log.info("Total CDF files to download: %d  (workers=%d)", total, workers)

    failures = []
    with tqdm(total=total, unit="file", desc=dataset_name, ncols=90) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_file, session, url, dest, pbar): (url, dest)
                for url, dest in tasks
            }
            for fut in as_completed(futures):
                ok, msg = fut.result()
                if not ok:
                    failures.append(msg)
                    log.error(msg)

    log.info("Finished %s — %d/%d succeeded", dataset_name, total - len(failures), total)
    if failures:
        log.warning("%d files failed:", len(failures))
        for f in failures:
            log.warning("  %s", f)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download NASA OMNI HRO CDF files in parallel."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Which dataset(s) to download (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallel download threads per dataset (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE_OUTPUT_DIR,
        help=f"Root output directory (default: {BASE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=str,
        default=None,
        help="Restrict download to specific years, e.g. --years 2020 2021 2022",
    )
    args = parser.parse_args()

    selected = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    # Monkey-patch year filter into collect_tasks if requested
    if args.years:
        year_set = set(args.years)
        original_list_links = list_links

        def filtered_list_links(session, url, suffix="/"):
            links = original_list_links(session, url, suffix)
            if suffix == "/":
                return [l for l in links if l.rstrip("/").split("/")[-1] in year_set]
            return links

        import __main__
        __main__.list_links = filtered_list_links

    for name in selected:
        download_dataset(
            dataset_name=name,
            base_url=DATASETS[name],
            output_dir=args.outdir / name,
            workers=args.workers,
        )

    log.info("All done.")


if __name__ == "__main__":
    main()