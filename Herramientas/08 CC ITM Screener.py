"""
get_r1000_tickers.py
=====================
Descarga los holdings del ETF iShares Russell 1000 (IWB) y extrae
la lista de tickers, sin depender de un parser XML estricto (el
archivo de BlackRock suele traer entidades/caracteres mal formados
que rompen xml.etree.ElementTree).
"""

import re
import requests

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

# Cada fila del XML de BlackRock viene como <ss:Row>...<ss:Cell><ss:Data ...>VALOR</ss:Data></ss:Cell>...</ss:Row>
ROW_RE = re.compile(r"<ss:Row[^>]*>(.*?)</ss:Row>", re.DOTALL)
CELL_RE = re.compile(r"<ss:Data[^>]*>(.*?)</ss:Data>", re.DOTALL)


def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().upper()
    # Quitamos posibles entidades HTML residuales (&amp; -> &, etc.)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    if not t or t in ("-", "NAN", "N/A"):
        return None
    return t.replace(" ", "")


def download_r1000_tickers():
    """Descarga y extrae los tickers de holdings de IWB (Russell 1000) vía regex."""
    resp = requests.get(IWB_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    text = resp.content.decode("utf-8", errors="ignore")

    rows = ROW_RE.findall(text)
    if not rows:
        return []

    # Localizamos la fila de cabecera (primera celda == "Ticker")
    header_row_idx = None
    for i, row in enumerate(rows):
        cells = CELL_RE.findall(row)
        if cells and cells[0].strip().lower() == "ticker":
            header_row_idx = i
            break

    if header_row_idx is None:
        return []

    tickers = []
    for row in rows[header_row_idx + 1:]:
        cells = CELL_RE.findall(row)
        if not cells:
            continue
        t = _clean_ticker(cells[0])
        if t:
            tickers.append(t)

    return sorted(set(tickers))


if __name__ == "__main__":
    tickers = download_r1000_tickers()
    print(f"Total tickers descargados: {len(tickers)}")
    print(tickers[:20])

    with open("r1000_tickers.csv", "w") as f:
        f.write("Ticker\n")
        for t in tickers:
            f.write(t + "\n")
