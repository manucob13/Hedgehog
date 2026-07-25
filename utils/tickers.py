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
# 5. RUSSELL 1000 (DESCARGA DINÁMICA DESDE WIKIPEDIA + CACHE)
# ============================================================

_RUSSELL1000_CACHE_FILE = "russell1000_cache.csv"


def get_russell1000_tickers(use_cache=True, cache_days=7):
    """
    Descarga la lista de tickers del Russell 1000 desde Wikipedia.
    Usa un cache local en CSV para no golpear Wikipedia en cada ejecución
    (importante porque este módulo se usa dentro de una app Streamlit
    con recargas frecuentes).

    Args:
        use_cache (bool): Si True, intenta usar el cache local antes de descargar
        cache_days (int): Días de validez del cache antes de refrescar

    Returns:
        list: Lista de tickers del Russell 1000 (formato compatible Yahoo Finance,
              ej. 'BRK.B' -> 'BRK-B'). Lista vacía si falla la descarga y no hay cache.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(current_dir, _RUSSELL1000_CACHE_FILE)

    # 1. Intentar usar cache si es reciente
    if use_cache and os.path.exists(cache_path):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - mtime).days < cache_days:
                cached = pd.read_csv(cache_path)
                tickers = cached["Ticker"].dropna().tolist()
                print(f"✅ Russell 1000: {len(tickers)} tickers cargados desde cache")
                return tickers
        except Exception:
            pass

    # 2. Descargar desde Wikipedia
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        # Buscamos la tabla de componentes por sus columnas, no por índice fijo,
        # ya que Wikipedia puede reordenar las tablas de la página.
        df = None
        for t in tables:
            if "Symbol" in t.columns and "Company" in t.columns:
                df = t
                break

        if df is None:
            raise ValueError(
                "No se encontró la tabla de componentes con columnas 'Company'/'Symbol'"
            )

        # Yahoo Finance usa '-' en vez de '.' para clases de acciones (ej: BRK.B -> BRK-B)
        tickers = (
            df["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        tickers = sorted(set(tickers))

        # 3. Guardar en cache
        try:
            pd.DataFrame({"Ticker": tickers}).to_csv(cache_path, index=False)
        except Exception as e:
            print(f"⚠️ No se pudo guardar cache de Russell 1000 ({e})")

        print(f"✅ Russell 1000: {len(tickers)} tickers descargados de Wikipedia")
        return tickers

    except Exception as e:
        print(f"⚠️ Error descargando Russell 1000 ({e}), usando cache si existe...")
        if os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                tickers = cached["Ticker"].dropna().tolist()
                print(f"✅ Russell 1000: {len(tickers)} tickers cargados desde cache (fallback)")
                return tickers
            except Exception:
                pass
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
