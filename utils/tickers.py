"""
download_tickers.py

Descarga y combina:
- Russell 1000 (top 1000 acciones USA por capitalización)
- Top Índices por volumen de opciones
- Top ETFs por volumen de opciones

Guarda todo en Tks.csv en el directorio actual

NUEVAS FUNCIONES PÚBLICAS:
- get_all_etf_tickers(): Retorna lista de todos los tickers de ETFs
- get_all_index_tickers(): Retorna lista de todos los tickers de índices
- get_etf_name(ticker): Retorna el nombre de un ETF específico
- get_index_name(ticker): Retorna el nombre de un índice específico
- get_all_etf_names(): Retorna diccionario completo {ticker: nombre} de ETFs
- get_all_index_names(): Retorna diccionario completo {ticker: nombre} de índices
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import os


def download_russell1000():
    """
    Descarga los ~1000 tickers del Russell 1000 desde iShares ETF (IWB)
    GRATIS - No requiere API key
    
    Returns:
        list: Lista de tickers únicos del Russell 1000
    """
    print("\n📥 Descargando Russell 1000 desde iShares (IWB)...")
    
    try:
        # URL del CSV de holdings de iShares Russell 1000 ETF
        url = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax"
        
        params = {
            'fileType': 'csv',
            'fileName': 'IWB_holdings',
            'dataType': 'fund'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Descargar con timeout de 30 segundos
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Decodificar contenido (UTF-8 con BOM)
        content = response.content.decode('utf-8-sig')
        
        # Procesar líneas
        lines = content.split('\n')
        
        # Encontrar la línea de headers (contiene 'Ticker')
        header_idx = None
        for i, line in enumerate(lines):
            if 'Ticker' in line and 'Name' in line:
                header_idx = i
                break
        
        if header_idx is None:
            raise ValueError("❌ No se encontró la línea de headers en el CSV")
        
        # Extraer datos desde la línea de headers hasta 'Total'
        data_lines = [lines[header_idx]]  # Header
        
        for line in lines[header_idx + 1:]:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('Total'):
                data_lines.append(line)
            elif line_stripped.startswith('Total'):
                break  # Fin de datos
        
        # Crear DataFrame desde las líneas procesadas
        df = pd.read_csv(StringIO('\n'.join(data_lines)))
        
        # Limpiar y filtrar tickers válidos
        df = df[df['Ticker'].notna()]  # Eliminar NaN
        df = df[df['Ticker'] != '-']   # Eliminar guiones
        
        # Filtrar solo tickers que sean letras mayúsculas (sin números ni símbolos)
        df = df[df['Ticker'].str.match(r'^[A-Z]+$', na=False)]
        
        # Obtener lista única de tickers y ordenar alfabéticamente
        tickers = sorted(df['Ticker'].unique().tolist())
        
        print(f"✅ Russell 1000: {len(tickers)} tickers descargados")
        
        return tickers
        
    except requests.exceptions.Timeout:
        print("❌ Error: Timeout al conectar con iShares. Intenta de nuevo.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return []
    except Exception as e:
        print(f"❌ Error procesando Russell 1000: {e}")
        return []


def get_all_index_names():
    """
    Retorna diccionario completo con nombres de índices
    
    Returns:
        dict: Diccionario {ticker: nombre_completo}
    """
    index_names = {
        "SPX": "S&P 500 Index",
        "VIX": "CBOE Volatility Index",
        "XSP": "Mini S&P 500",
        "RUT": "Russell 2000",
        "NDX": "Nasdaq 100",
        "DJX": "Dow Jones",
        "NANOS": "Nano S&P 500",
        "OEX": "S&P 100",
        "XEO": "S&P 100 European",
        "MXEF": "MSCI Emerging Markets",
        "MXEA": "MSCI EAFE",
    }
    return index_names


def get_all_etf_names():
    """
    Retorna diccionario completo con nombres de ETFs
    
    Returns:
        dict: Diccionario {ticker: nombre_completo}
    """
    etf_names = {
        # MEGA CAPS
        "SPY": "SPDR S&P 500",
        "QQQ": "Invesco QQQ",
        "IWM": "iShares Russell 2000",
        
        # LEVERAGED/INVERSE
        "TQQQ": "ProShares UltraPro QQQ 3x",
        "SQQQ": "ProShares UltraPro Short QQQ -3x",
        
        # FIXED INCOME & COMMODITIES
        "TLT": "iShares 20+ Year Treasury",
        "GLD": "SPDR Gold Shares",
        "HYG": "iShares High Yield Corporate Bond",
        "UNG": "United States Natural Gas Fund",
        "USO": "United States Oil Fund",
        "SLV": "iShares Silver Trust",
        "LQD": "iShares Investment Grade Corporate",
        
        # INTERNATIONAL
        "EEM": "iShares MSCI Emerging Markets",
        "FXI": "iShares China Large-Cap",
        "EFA": "iShares MSCI EAFE",
        "KWEB": "KraneShares CSI China Internet",
        "EWZ": "iShares MSCI Brazil",
        "EWJ": "iShares MSCI Japan",
        "EWU": "iShares MSCI United Kingdom",
        "EWG": "iShares MSCI Germany",
        "EWC": "iShares MSCI Canada",
        
        # SECTOR SPDR
        "XLF": "Financial Select Sector",
        "XLE": "Energy Select Sector",
        "XLI": "Industrial Select Sector",
        "XLK": "Technology Select Sector",
        "XLV": "Health Care Select Sector",
        "XLU": "Utilities Select Sector",
        "XLP": "Consumer Staples Select Sector",
        "XLY": "Consumer Discretionary Select Sector",
        "XLB": "Materials Select Sector",
        "XLRE": "Real Estate Select Sector",
        "XLC": "Communication Services Select Sector",
        
        # SPECIALIZED SECTORS
        "GDX": "VanEck Gold Miners",
        "XBI": "SPDR Biotech",
        "SMH": "VanEck Semiconductor",
        "XOP": "SPDR Oil & Gas Exploration",
        "XRT": "SPDR Retail",
        "XHB": "SPDR Homebuilders",
        "XME": "SPDR Metals & Mining",
        "GDXJ": "VanEck Junior Gold Miners",
        "OIH": "VanEck Oil Services",
        
        # VOLATILITY
        "VXX": "iPath Series B S&P 500 VIX",
        "UVXY": "ProShares Ultra VIX Short-Term",
        "SVXY": "ProShares Short VIX Short-Term",
        
        # THEMATIC & SPECIALIZED
        "DIA": "SPDR Dow Jones Industrial Average",
        "BITO": "ProShares Bitcoin Strategy",
        "ARKK": "ARK Innovation ETF",
        "JETS": "U.S. Global Jets",
        "MSOS": "AdvisorShares Pure US Cannabis",
        "SOXX": "iShares Semiconductor",
        
        # LEVERAGED SPECIALIZED
        "LABU": "Direxion Daily S&P Biotech Bull 3X",
        "BOIL": "ProShares Ultra Bloomberg Natural Gas 2x",
        "TNA": "Direxion Daily Small Cap Bull 3X",
        "SPXS": "Direxion Daily S&P 500 Bear -3X",
        "SPXU": "ProShares UltraPro Short S&P 500",
        "SOXS": "Direxion Daily Semiconductor Bear 3X",
        "TZA": "Direxion Daily Small Cap Bear 3X",
        "TMF": "Direxion Daily 20+ Year Treasury Bull 3X",
        "TSLL": "Direxion Daily TSLA Bull 1.5X",
        
        # ADDITIONAL LIQUID ETFs
        "IYR": "iShares U.S. Real Estate",
        "ASHR": "Xtrackers Harvest CSI 300 China",
        "UUP": "Invesco DB US Dollar Index Bullish",
    }
    return etf_names


def get_index_name(ticker):
    """
    Obtiene el nombre completo de un índice dado su ticker
    
    Args:
        ticker (str): Símbolo del índice (ej: 'SPX', 'VIX')
    
    Returns:
        str: Nombre completo del índice, o None si no se encuentra
    
    Ejemplo:
        >>> get_index_name('SPX')
        'S&P 500 Index'
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
    
    Ejemplo:
        >>> get_etf_name('SPY')
        'SPDR S&P 500'
    """
    etf_names = get_all_etf_names()
    return etf_names.get(ticker.upper())


def get_all_index_tickers():
    """
    Retorna lista de todos los tickers de índices disponibles
    
    Returns:
        list: Lista de símbolos de índices
    
    Ejemplo:
        >>> tickers = get_all_index_tickers()
        >>> print(tickers[:3])
        ['SPX', 'VIX', 'XSP']
    """
    return list(get_all_index_names().keys())


def get_all_etf_tickers():
    """
    Retorna lista de todos los tickers de ETFs disponibles
    
    Returns:
        list: Lista de símbolos de ETFs
    
    Ejemplo:
        >>> tickers = get_all_etf_tickers()
        >>> print(len(tickers))
        72
    """
    return list(get_all_etf_names().keys())


def get_top_indices():
    """
    Retorna lista de índices con mayor volumen de opciones
    Basado en datos de CBOE 2023-2024
    
    Returns:
        list: Lista de símbolos de índices
    """
    indices = get_all_index_tickers()
    print(f"✅ Índices: {len(indices)} símbolos agregados")
    return indices


def get_top_etfs():
    """
    Retorna lista de ETFs con mayor volumen de opciones
    Ordenados por volumen promedio diario de contratos (2023-2024)
    
    Returns:
        list: Lista de símbolos de ETFs
    """
    etfs = get_all_etf_tickers()
    print(f"✅ ETFs: {len(etfs)} símbolos agregados")
    return etfs


def create_tickers_universe(output_filename='Tks.csv'):
    """
    Crea el universo completo de tickers combinando:
    - Russell 1000 (acciones)
    - Top Índices por volumen de opciones
    - Top ETFs por volumen de opciones
    
    Guarda el resultado en un archivo CSV en el directorio actual
    
    Args:
        output_filename (str): Nombre del archivo CSV de salida
    
    Returns:
        pd.DataFrame: DataFrame con todos los tickers
    """
    print("=" * 70)
    print("🚀 DESCARGANDO UNIVERSO COMPLETO DE TICKERS")
    print("=" * 70)
    
    # 1. Descargar Russell 1000
    stocks = download_russell1000()
    stocks_df = pd.DataFrame({
        'Ticker': stocks,
        'Type': 'Stock'
    })
    
    # 2. Obtener índices
    indices = get_top_indices()
    indices_df = pd.DataFrame({
        'Ticker': indices,
        'Type': 'Index'
    })
    
    # 3. Obtener ETFs
    etfs = get_top_etfs()
    etfs_df = pd.DataFrame({
        'Ticker': etfs,
        'Type': 'ETF'
    })
    
    # 4. Combinar todos
    all_df = pd.concat([stocks_df, indices_df, etfs_df], ignore_index=True)
    
    # 5. Eliminar duplicados (por si acaso)
    all_df = all_df.drop_duplicates(subset='Ticker', keep='first')
    
    # 6. Ordenar alfabéticamente
    all_df = all_df.sort_values('Ticker').reset_index(drop=True)
    
    # 7. Agregar metadata
    all_df['LastUpdate'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 8. Guardar en el directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)
    
    all_df.to_csv(output_path, index=False)
    
    # 9. Estadísticas finales
    print("\n" + "=" * 70)
    print("📊 RESUMEN DEL UNIVERSO DE TICKERS")
    print("=" * 70)
    print(f"📈 Acciones (Russell 1000): {len(stocks_df):,}")
    print(f"📉 Índices: {len(indices_df):,}")
    print(f"📊 ETFs: {len(etfs_df):,}")
    print("-" * 70)
    print(f"🎯 TOTAL DE TICKERS: {len(all_df):,}")
    print("=" * 70)
    print(f"\n💾 Archivo guardado en: {output_path}")
    print(f"📅 Fecha de actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Proceso completado exitosamente!")
    
    # 10. Mostrar muestra
    print("\n" + "=" * 70)
    print("📋 MUESTRA DE DATOS (primeros 20 registros):")
    print("=" * 70)
    print(all_df.head(20).to_string(index=False))
    
    # Mostrar distribución por tipo
    print("\n" + "=" * 70)
    print("📊 DISTRIBUCIÓN POR TIPO:")
    print("=" * 70)
    type_counts = all_df['Type'].value_counts()
    for ticker_type, count in type_counts.items():
        percentage = (count / len(all_df)) * 100
        print(f"{ticker_type:10s}: {count:5,} ({percentage:5.1f}%)")
    
    return all_df


def main():
    """
    Función principal para ejecutar el script
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
    # Ejecutar descarga
    result_df = main()
    
    if result_df is not None:
        print("\n🎉 ¡Descarga completada con éxito!")
    else:
        print("\n⚠️ La descarga falló. Revisa los errores arriba.")
