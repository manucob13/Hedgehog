"""
tickers.py

Universo FIJO de tickers para el Trend Stocks Screener.

Contiene una lista curada de 210 activos de alta liquidez:
- 5 Índices
- 72 ETFs
- 133 Acciones (Semiconductores, Big Tech, Financiero/Fintech/Cripto,
  Energía/Industria/Aeroespacial, Consumo, Salud/Biotech, Telecom/Meme/ADRs)

Además, incorpora la descarga dinámica (con cache local) del universo
Russell 1000 desde Wikipedia, para ampliar el screener a ese índice.

FUNCIONES PÚBLICAS (mismas firmas que antes, compatibles con app.py):
- create_tickers_universe(): Retorna pd.DataFrame con todo el universo
- get_all_etf_tickers(): Retorna lista de todos los tickers de ETFs
- get_all_index_tickers(): Retorna lista de todos los tickers de índices
- get_etf_name(ticker): Retorna el nombre de un ETF específico
- get_index_name(ticker): Retorna el nombre de un índice específico
- get_all_etf_names(): Retorna diccionario completo {ticker: nombre} de ETFs
- get_all_index_names(): Retorna diccionario completo {ticker: nombre} de índices
- get_stock_tickers(): Retorna lista de todos los tickers de acciones
- get_russell1000_tickers(): Retorna lista de tickers del Russell 1000 (Wikipedia + cache)
"""

import os
import pandas as pd
import requests
from io import StringIO
from datetime import datetime


# ============================================================
# 1. ÍNDICES (5)
# ============================================================

def get_all_index_names():
    """
    Retorna diccionario completo con nombres de índices

    Returns:
        dict: Diccionario {ticker: nombre_completo}
    """
    index_names = {
        "SPX": "S&P 500 Index",
        "NDX": "Nasdaq 100 Index",
        "VIX": "CBOE Volatility Index",
        "RUT": "Russell 2000 Index",
        "XSP": "Mini S&P 500 Index",
    }
    return index_names


# ============================================================
# 2. ETFs (72)
# ============================================================

def get_all_etf_names():
    """
    Retorna diccionario completo con nombres de ETFs

    Returns:
        dict: Diccionario {ticker: nombre_completo}
    """
    etf_names = {
        # --- Mercado General (17) ---
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "IWM": "iShares Russell 2000 ETF",
        "DIA": "SPDR Dow Jones Industrial Average ETF",
        "ONEQ": "Fidelity Nasdaq Composite ETF",
        "VOO": "Vanguard S&P 500 ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "MDY": "SPDR S&P MidCap 400 ETF",
        "SPLG": "SPDR Portfolio S&P 500 ETF",
        "VT": "Vanguard Total World Stock ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "OEF": "iShares S&P 100 ETF",
        "RSP": "Invesco S&P 500 Equal Weight ETF",
        "VIXY": "ProShares VIX Short-Term Futures ETF",
        "UVXY": "ProShares Ultra VIX Short-Term Futures ETF",
        "VXX": "iPath Series B S&P 500 VIX Short-Term Futures ETN",
        "SVXY": "ProShares Short VIX Short-Term Futures ETF",

        # --- Semiconductores / Hardware (5) ---
        "SOXL": "Direxion Daily Semiconductor Bull 3X",
        "SOXS": "Direxion Daily Semiconductor Bear 3X",
        "SMH": "VanEck Semiconductor ETF",
        "SOXX": "iShares Semiconductor ETF",
        "NVDL": "GraniteShares 2x Long NVDA Daily ETF",

        # --- Sectoriales (21) ---
        "XLF": "Financial Select Sector SPDR",
        "XLE": "Energy Select Sector SPDR",
        "XLK": "Technology Select Sector SPDR",
        "KRE": "SPDR S&P Regional Banking ETF",
        "XBI": "SPDR S&P Biotech ETF",
        "XLV": "Health Care Select Sector SPDR",
        "XLY": "Consumer Discretionary Select Sector SPDR",
        "XLP": "Consumer Staples Select Sector SPDR",
        "XLI": "Industrial Select Sector SPDR",
        "XLB": "Materials Select Sector SPDR",
        "XLU": "Utilities Select Sector SPDR",
        "XLRE": "Real Estate Select Sector SPDR",
        "IYR": "iShares U.S. Real Estate ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "IBB": "iShares Biotechnology ETF",
        "ITB": "iShares U.S. Home Construction ETF",
        "XHB": "SPDR S&P Homebuilders ETF",
        "XOP": "SPDR S&P Oil & Gas Exploration & Production ETF",
        "OIH": "VanEck Oil Services ETF",
        "XRT": "SPDR S&P Retail ETF",
        "IYT": "iShares U.S. Transportation ETF",

        # --- Cripto (3) ---
        "IBIT": "iShares Bitcoin Trust",
        "FBTC": "Fidelity Wise Origin Bitcoin Fund",
        "GBTC": "Grayscale Bitcoin Trust",

        # --- Apalancado individual (1) ---
        "TSLL": "Direxion Daily TSLA Bull 2X Shares",

        # --- Internacionales / Materias Primas (14) ---
        "FXI": "iShares China Large-Cap ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "EWZ": "iShares MSCI Brazil ETF",
        "ASHR": "Xtrackers Harvest CSI 300 China A-Shares ETF",
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "GDX": "VanEck Gold Miners ETF",
        "GDXJ": "VanEck Junior Gold Miners ETF",
        "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund",
        "EFA": "iShares MSCI EAFE ETF",
        "INDA": "iShares MSCI India ETF",
        "EWJ": "iShares MSCI Japan ETF",
        "KWEB": "KraneShares CSI China Internet ETF",

        # --- Renta Fija y Apalancados Mercado General (11) ---
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "HYG": "iShares iBoxx High Yield Corporate Bond ETF",
        "LQD": "iShares iBoxx Investment Grade Corporate Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "SHY": "iShares 1-3 Year Treasury Bond ETF",
        "TQQQ": "ProShares UltraPro QQQ 3x",
        "SQQQ": "ProShares UltraPro Short QQQ -3x",
        "SPXL": "Direxion Daily S&P 500 Bull 3X",
        "SPXS": "Direxion Daily S&P 500 Bear 3X",
        "UPRO": "ProShares UltraPro S&P 500 3x",
        "TMF": "Direxion Daily 20+ Year Treasury Bull 3X",
    }
    return etf_names


# ============================================================
# 3. ACCIONES (133)
# ============================================================

def get_stock_tickers():
    """
    Retorna lista fija de acciones del universo.
    NOTA: TSMC se mapea al ticker real 'TSM' (Yahoo Finance no reconoce 'TSMC').

    Returns:
        list: Lista de tickers de acciones
    """
    stocks = [
        # --- Semiconductores, Hardware y Supercomputación (22) ---
        "NVDA", "AMD", "INTC", "SMCI", "AVGO", "MU", "ARM", "TSM", "QCOM",
        "ASML", "AMAT", "LRCX", "ADI", "NXPI", "TXN", "KLAC", "MRVL",
        "DELL", "HPQ", "HPE", "WDC", "STX",

        # --- Big Tech, Software, IA y Cloud (25) ---
        "AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG", "PLTR", "NFLX",
        "CRM", "ORCL", "NOW", "PANW", "CSCO", "IBM", "UBER", "PATH",
        "SNOW", "NET", "WDAY", "DDOG", "CRWD", "ZS", "ADBE", "TEAM", "MSI",

        # --- Financiero, Fintech y Cripto-Activos (23) ---
        "BAC", "JPM", "GS", "MS", "C", "WFC", "V", "MA", "AXP", "COIN",
        "MSTR", "HOOD", "PYPL", "SQ", "SOFI", "MARA", "RIOT", "SCHW",
        "BLK", "NU", "AFRM", "AIG", "MET",

        # --- Energía, Automoción, Industria y Aeroespacial (23) ---
        "TSLA", "F", "GM", "XOM", "CVX", "GE", "BA", "CAT", "FDX", "UPS",
        "RIVN", "LCID", "NIO", "COP", "SLB", "HAL", "RTX", "LMT", "DE",
        "MMM", "HON", "UNP", "FCX",

        # --- Consumo Masivo, Retail y Entretenimiento (20) ---
        "WMT", "TGT", "HD", "NKE", "DIS", "SBUX", "KO", "PEP", "COST",
        "MCD", "CMG", "LOW", "TJX", "EL", "CL", "PGR", "CVS", "WBA",
        "LVS", "MGM",

        # --- Salud, Farmacéuticas y Biotecnología (13) ---
        "LLY", "PFE", "UNH", "JNJ", "ABBV", "MRK", "AMGN", "GILD", "BMY",
        "MRNA", "BIIB", "VRTX", "REGN",

        # --- Telecom, Meme Stocks y ADRs especulativos (7) ---
        "T", "VZ", "GME", "AMC", "BABA", "PDD", "JD",
    ]
    return stocks


# ============================================================
# 4. FUNCIONES DE ACCESO (mismo interfaz que antes)
# ============================================================

def get_index_name(ticker):
    """
    Obtiene el nombre completo de un índice dado su ticker

    Args:
        ticker (str): Símbolo del índice (ej: 'SPX', 'VIX')

    Returns:
        str: Nombre completo del índice, o None si no se encuentra
    """
    index_names = get_all_index_names()
    return index_names.get(ticker.upper())


def get_etf_name(ticker):
    """
    Obtiene el nombre completo de un ETF dado su ticker

    Args:
        ticker (str): Símbolo del ETF (ej: 'SPY', 'QQQ')

    Returns:
        str: Nombre completo del ETF, o None si no se encuentra
    """
    etf_names = get_all_etf_names()
    return etf_names.get(ticker.upper())


def get_all_index_tickers():
    """
    Retorna lista de todos los tickers de índices disponibles

    Returns:
        list: Lista de símbolos de índices
    """
    return list(get_all_index_names().keys())


def get_all_etf_tickers():
    """
    Retorna lista de todos los tickers de ETFs disponibles

    Returns:
        list: Lista de símbolos de ETFs
    """
    return list(get_all_etf_names().keys())


def get_top_indices():
    """
    Retorna lista de índices del universo fijo

    Returns:
        list: Lista de símbolos de índices
    """
    indices = get_all_index_tickers()
    print(f"✅ Índices: {len(indices)} símbolos agregados")
    return indices


def get_top_etfs():
    """
    Retorna lista de ETFs del universo fijo

    Returns:
        list: Lista de símbolos de ETFs
    """
    etfs = get_all_etf_tickers()
    print(f"✅ ETFs: {len(etfs)} símbolos agregados")
    return etfs


def get_top_stocks():
    """
    Retorna lista de acciones del universo fijo

    Returns:
        list: Lista de símbolos de acciones
    """
    stocks = get_stock_tickers()
    print(f"✅ Acciones: {len(stocks)} símbolos agregados")
    return stocks

# ============================================================
# 5. RUSSELL 1000 (IWB HOLDINGS CSV + WIKIPEDIA FALLBACK + CACHE)
# ============================================================
#
# CAMBIO respecto a la versión anterior: Wikipedia cambió el formato de
# la tabla de componentes de la página del Russell 1000 (ya no expone
# una columna con el texto literal "Symbol"), lo que rompía el parser
# de pd.read_html y hacía caer el universo a 0 tickers de Russell 1000
# en cuanto expiraba (o desaparecía) el cache local.
#
# Nueva estrategia, en orden:
#   1. CSV oficial de holdings del ETF iShares Russell 1000 (IWB) —
#      fuente primaria del propio índice, mucho más estable que
#      scrapear HTML de Wikipedia.
#   2. Wikipedia como fallback, con detección de columnas tolerante
#      (Symbol/Ticker, case-insensitive) en vez de un match exacto.
#   3. Cache local en CSV como último recurso si ambas fuentes fallan
#      (por ejemplo, sin conexión).
#
# NOTA IMPORTANTE si despliegas en Streamlit Cloud (o cualquier entorno
# con filesystem efímero): el cache local se borra en cada redeploy o
# reinicio del contenedor, así que NUNCA debe ser tu única red de
# seguridad — de ahí que ahora haya dos fuentes en vivo antes de caer
# al cache.

_RUSSELL1000_CACHE_FILE = "russell1000_cache.csv"

_IWB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}


def _clean_russell_ticker(raw):
    """Normaliza un ticker crudo (de iShares o Wikipedia) al formato que
    usa Yahoo Finance: mayúsculas, sin espacios, '.' -> '-' para clases
    de acciones (ej. BRK.B -> BRK-B). Devuelve None si no parece un
    ticker válido (filas de cash, totales, texto de disclaimer, etc.)."""
    if raw is None:
        return None
    t = str(raw).strip().upper().replace(".", "-")
    if not t or t in ("-", "NAN", "N/A", "CASH", "USD", "NET", "OTHER"):
        return None
    # las filas de cash/derivados en el CSV de iShares suelen venir sin
    # letras (ej. '-', '--') o con caracteres no alfabéticos
    if not any(c.isalpha() for c in t):
        return None
    return t


def _read_russell1000_from_ishares():
    """Descarga y parsea el CSV de holdings de IWB. El archivo trae ~9
    líneas de metadata antes de la cabecera real y unas líneas de
    disclaimer al final, así que se detecta la fila de cabecera
    buscando la palabra 'Ticker' en vez de asumir un skiprows fijo
    (el número exacto de filas de metadata varía con el tiempo)."""
    resp = requests.get(_IWB_HOLDINGS_URL, headers=_REQUEST_HEADERS, timeout=20)
    resp.raise_for_status()
    raw_lines = resp.text.splitlines()

    header_idx = None
    for i, line in enumerate(raw_lines[:40]):
        if line.strip().lower().startswith("ticker,") or ",ticker," in line.lower() or line.strip().lower().startswith('"ticker"'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No se encontró la fila de cabecera con 'Ticker' en el CSV de IWB")

    csv_body = "\n".join(raw_lines[header_idx:])
    df = pd.read_csv(StringIO(csv_body), on_bad_lines="skip")

    ticker_col = None
    for c in df.columns:
        if str(c).strip().lower() == "ticker":
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("El CSV de IWB no tiene columna 'Ticker' tras parsear")

    tickers = [_clean_russell_ticker(t) for t in df[ticker_col].tolist()]
    tickers = sorted(set(t for t in tickers if t))
    if len(tickers) < 500:  # sanity check — el Russell 1000 real ronda ~1000
        raise ValueError(f"Solo {len(tickers)} tickers válidos extraídos del CSV de IWB — probablemente mal parseado")
    return tickers


def _read_russell1000_from_wikipedia():
    """Fallback: scrape de Wikipedia con detección de columnas tolerante
    (Symbol o Ticker, sin importar mayúsculas/minúsculas), ya que el
    nombre exacto de la columna ha cambiado antes sin aviso."""
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    df = None
    ticker_col = None
    for t in tables:
        cols_lower = {str(c).strip().lower(): c for c in t.columns}
        for candidate in ("symbol", "ticker"):
            if candidate in cols_lower and len(t) > 500:
                df = t
                ticker_col = cols_lower[candidate]
                break
        if df is not None:
            break

    if df is None:
        raise ValueError("No se encontró en Wikipedia ninguna tabla >500 filas con columna Symbol/Ticker")

    tickers = [_clean_russell_ticker(t) for t in df[ticker_col].tolist()]
    tickers = sorted(set(t for t in tickers if t))
    if len(tickers) < 500:
        raise ValueError(f"Solo {len(tickers)} tickers válidos extraídos de Wikipedia — probablemente mal parseado")
    return tickers


def get_russell1000_tickers(use_cache=True, cache_days=7):
    """
    Obtiene la lista de tickers del Russell 1000. Orden de fuentes:
    1) CSV oficial de holdings de IWB (iShares Russell 1000 ETF)
    2) Wikipedia (fallback si iShares falla)
    3) Cache local en CSV (último recurso, solo si ambas fuentes en
       vivo fallan)

    Args:
        use_cache (bool): si True, intenta servir desde cache reciente
            ANTES de golpear la red (ahorra llamadas en recargas
            frecuentes de Streamlit)
        cache_days (int): días de validez del cache antes de refrescar

    Returns:
        list: tickers del Russell 1000 en formato Yahoo Finance
              (ej. 'BRK.B' -> 'BRK-B'). Lista vacía solo si TODO falla
              y tampoco hay cache utilizable.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, _RUSSELL1000_CACHE_FILE)

    if use_cache and os.path.exists(cache_path):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - mtime).days < cache_days:
                cached = pd.read_csv(cache_path)
                tickers = cached["Ticker"].dropna().tolist()
                if len(tickers) >= 500:
                    print(f"✅ Russell 1000: {len(tickers)} tickers cargados desde cache")
                    return tickers
        except Exception:
            pass

    last_error = None
    for source_name, source_fn in (
        ("iShares (IWB)", _read_russell1000_from_ishares),
        ("Wikipedia", _read_russell1000_from_wikipedia),
    ):
        try:
            tickers = source_fn()
            try:
                pd.DataFrame({"Ticker": tickers}).to_csv(cache_path, index=False)
            except Exception as e:
                print(f"⚠️ No se pudo guardar cache de Russell 1000 ({e})")
            print(f"✅ Russell 1000: {len(tickers)} tickers descargados desde {source_name}")
            return tickers
        except Exception as e:
            last_error = e
            print(f"⚠️ Fuente '{source_name}' falló para Russell 1000: {e}")
            continue

    print(f"⚠️ Todas las fuentes en vivo fallaron ({last_error}), usando cache si existe...")
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path)
            tickers = cached["Ticker"].dropna().tolist()
            print(f"✅ Russell 1000: {len(tickers)} tickers cargados desde cache (fallback final)")
            return tickers
        except Exception:
            pass

    print("❌ Russell 1000: no se pudo obtener de ninguna fuente ni del cache")
    return []


def get_top_russell1000():
    """
    Retorna lista de tickers del Russell 1000 (wrapper con log estándar,
    siguiendo el mismo estilo que get_top_indices/get_top_etfs/get_top_stocks).

    Returns:
        list: Lista de símbolos del Russell 1000
    """
    russell = get_russell1000_tickers()
    print(f"✅ Russell 1000: {len(russell)} símbolos agregados")
    return russell


# ============================================================
# 6. CONSTRUCCIÓN DEL UNIVERSO
# ============================================================

def create_tickers_universe(output_filename='Tks.csv', include_russell1000=True):
    """
    Crea el universo completo de tickers combinando la lista fija de:
    - Acciones (Semiconductores, Big Tech, Financiero, Energía/Industria,
      Consumo, Salud, Telecom/Meme)
    - Índices
    - ETFs
    - (Opcional) Russell 1000, descargado dinámicamente desde Wikipedia con cache

    Guarda el resultado en un archivo CSV en el directorio actual.

    Args:
        output_filename (str): Nombre del archivo CSV de salida
        include_russell1000 (bool): Si True, incorpora el universo Russell 1000

    Returns:
        pd.DataFrame: DataFrame con todos los tickers
    """
    print("=" * 70)
    print("🚀 CONSTRUYENDO UNIVERSO FIJO DE TICKERS")
    print("=" * 70)

    # 1. Acciones
    stocks = get_top_stocks()
    stocks_df = pd.DataFrame({
        'Ticker': stocks,
        'Type': 'Stock'
    })

    # 2. Índices
    indices = get_top_indices()
    indices_df = pd.DataFrame({
        'Ticker': indices,
        'Type': 'Index'
    })

    # 3. ETFs
    etfs = get_top_etfs()
    etfs_df = pd.DataFrame({
        'Ticker': etfs,
        'Type': 'ETF'
    })

    dfs_to_concat = [stocks_df, indices_df, etfs_df]

    # 4. Russell 1000 (opcional, dinámico)
    if include_russell1000:
        russell = get_top_russell1000()
        if russell:
            russell_df = pd.DataFrame({
                'Ticker': russell,
                'Type': 'Russell1000'
            })
            dfs_to_concat.append(russell_df)
        else:
            print("⚠️ Russell 1000 no disponible, se omite del universo.")

    # 5. Combinar todos
    all_df = pd.concat(dfs_to_concat, ignore_index=True)

    # 6. Eliminar duplicados (por si acaso, priorizando la primera aparición:
    #    Stock > Index > ETF > Russell1000)
    all_df = all_df.drop_duplicates(subset='Ticker', keep='first')

    # 7. Ordenar alfabéticamente
    all_df = all_df.sort_values('Ticker').reset_index(drop=True)

    # 8. Agregar metadata
    all_df['LastUpdate'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 9. Guardar en el directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)

    try:
        all_df.to_csv(output_path, index=False)
        print(f"\n💾 Archivo guardado en: {output_path}")
    except Exception as e:
        print(f"\n⚠️ No se pudo guardar el CSV ({e}), continuando en memoria...")

    # 10. Estadísticas finales
    n_russell = len(all_df[all_df['Type'] == 'Russell1000'])
    print("\n" + "=" * 70)
    print("📊 RESUMEN DEL UNIVERSO DE TICKERS")
    print("=" * 70)
    print(f"📈 Acciones: {len(stocks_df):,}")
    print(f"📉 Índices: {len(indices_df):,}")
    print(f"📊 ETFs: {len(etfs_df):,}")
    if include_russell1000:
        print(f"🧩 Russell 1000: {n_russell:,}")
    print("-" * 70)
    print(f"🎯 TOTAL DE TICKERS (sin duplicados): {len(all_df):,}")
    print("=" * 70)
    print(f"📅 Fecha de actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Proceso completado exitosamente!")

    return all_df


def main():
    """
    Función principal para ejecutar el script de forma independiente
    """
    try:
        df = create_tickers_universe(output_filename='Tks.csv')
        return df
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result_df = main()

    if result_df is not None:
        print("\n🎉 ¡Universo generado con éxito!")
    else:
        print("\n⚠️ La generación falló. Revisa los errores arriba.")
