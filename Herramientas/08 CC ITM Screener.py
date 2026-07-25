"""
get_r1000_tickers.py
=====================
Descarga los holdings del ETF iShares Russell 1000 (IWB) directamente
desde el endpoint de BlackRock y extrae la lista de tickers.

El archivo que devuelve BlackRock es un XML tipo "SpreadsheetML"
(formato antiguo de Excel), no un .xlsx real, por eso hay que
parsearlo como XML en vez de usar pandas.read_excel.
"""

import re
import requests
import xml.etree.ElementTree as ET

IWB_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
    "&locale=en_US&portfolioId=239707&component=fundDownload&userType=individual"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}


def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().upper()
    if not t or t in ("-", "NAN", "N/A"):
        return None
    return t.replace(" ", "")


def download_r1000_tickers():
    """Descarga y parsea los tickers de holdings de IWB (Russell 1000)."""
    resp = requests.get(IWB_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # El contenido es XML SpreadsheetML; limpiamos caracteres de
    # control inválidos que rompen el parser XML.
    text = resp.content.decode("utf-8", errors="ignore")
    text_clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    root = ET.fromstring(text_clean)
    ws = root.find("ss:Worksheet", NS)
    table = ws.find("ss:Table", NS)
    rows = table.findall("ss:Row", NS)

    def cell_text(cell):
        data = cell.find("ss:Data", NS)
        return data.text if data is not None else None

    # Buscamos la fila de cabecera real (la que empieza con "Ticker")
    header_idx = None
    for i, row in enumerate(rows):
        cells = row.findall("ss:Cell", NS)
        if cells and cell_text(cells[0]) and cell_text(cells[0]).strip().lower() == "ticker":
            header_idx = i
            break

    if header_idx is None:
        return []

    tickers = []
    for row in rows[header_idx + 1:]:
        cells = row.findall("ss:Cell", NS)
        if not cells:
            continue
        t = _clean_ticker(cell_text(cells[0]))
        if t:
            tickers.append(t)

    return sorted(set(tickers))


if __name__ == "__main__":
    tickers = download_r1000_tickers()
    print(f"Total tickers descargados: {len(tickers)}")
    print(tickers[:20])

    # Opcional: guardar a CSV
    with open("r1000_tickers.csv", "w") as f:
        f.write("Ticker\n")
        for t in tickers:
            f.write(t + "\n")
