"""
download_tickers.py

Descarga y combina:
- Russell 1000 (top 1000 acciones USA por capitalización)
- Top Índices por volumen de opciones
- Top ETFs por volumen de opciones

Guarda todo en Tks.csv en el directorio actual
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


def get_top_indices():
    """
    Retorna lista de índices con mayor volumen de opciones
    Basado en datos de CBOE 2023-2024
    
    Returns:
        list: Lista de símbolos de índices
    """
    indices = [
        "SPX",      # S&P 500 Index - 2.9M contratos/día
        "VIX",      # CBOE Volatility Index - 742K contratos/día
        "XSP",      # Mini S&P 500 - 64K contratos/día
        "RUT",      # Russell 2000 - 62K contratos/día
        "NDX",      # Nasdaq 100 - 7K contratos/día
        "DJX",      # Dow Jones - 5K contratos/día
        "NANOS",    # Nano S&P 500 - 2.4K contratos/día
        "OEX",      # S&P 100 - 121 contratos/día
        "XEO",      # S&P 100 European
        "MXEF",     # MSCI Emerging Markets
        "MXEA",     # MSCI EAFE
    ]
    
    print(f"✅ Índices: {len(indices)} símbolos agregados")
    return indices


def get_top_etfs():
    """
    Retorna lista de ETFs con mayor volumen de opciones
    Ordenados por volumen promedio diario de contratos (2023-2024)
    
    Returns:
        list: Lista de símbolos de ETFs
    """
    etfs = [
        # MEGA CAPS - Más de 100K contratos/día
        "SPY",      # SPDR S&P 500 - 2.8M contratos/día
        "QQQ",      # Invesco QQQ - 926K contratos/día
        "IWM",      # iShares Russell 2000 - 262K contratos/día
        
        # LEVERAGED/INVERSE - 50K-100K contratos/día
        "TQQQ",     # ProShares UltraPro QQQ 3x - 92K
        "SQQQ",     # ProShares UltraPro Short QQQ -3x - 63K
        
        # FIXED INCOME & COMMODITIES - 20K-50K contratos/día
        "TLT",      # iShares 20+ Year Treasury - 40K
        "GLD",      # SPDR Gold Shares - 35K
        "HYG",      # iShares High Yield Corporate Bond - 28K
        "UNG",      # United States Natural Gas Fund - 18K
        "USO",      # United States Oil Fund - 8K
        "SLV",      # iShares Silver Trust - 15K
        "LQD",      # iShares Investment Grade Corporate - 4K
        
        # INTERNATIONAL - 20K-50K contratos/día
        "EEM",      # iShares MSCI Emerging Markets - 34K
        "FXI",      # iShares China Large-Cap - 24K
        "EFA",      # iShares MSCI EAFE - 21K
        "KWEB",     # KraneShares CSI China Internet - 18K
        "EWZ",      # iShares MSCI Brazil - 12K
        "EWJ",      # iShares MSCI Japan - 10K
        "EWU",      # iShares MSCI United Kingdom - 5K
        "EWG",      # iShares MSCI Germany - 4K
        "EWC",      # iShares MSCI Canada - 3K
        
        # SECTOR SPDR - 15K-30K contratos/día
        "XLF",      # Financial Select Sector - 27K
        "XLE",      # Energy Select Sector - 26K
        "XLI",      # Industrial Select Sector - 17K
        "XLK",      # Technology Select Sector - 3K
        "XLV",      # Health Care Select Sector - 3K
        "XLU",      # Utilities Select Sector - 8K
        "XLP",      # Consumer Staples Select Sector - 4K
        "XLY",      # Consumer Discretionary Select Sector - 5K
        "XLB",      # Materials Select Sector - 2K
        "XLRE",     # Real Estate Select Sector - 1K
        "XLC",      # Communication Services Select Sector - 1K
        
        # SPECIALIZED SECTORS - 10K-20K contratos/día
        "GDX",      # VanEck Gold Miners - 29K
        "XBI",      # SPDR Biotech - 14K
        "SMH",      # VanEck Semiconductor - 10K
        "XOP",      # SPDR Oil & Gas Exploration - 7K
        "XRT",      # SPDR Retail - 3K
        "XHB",      # SPDR Homebuilders - 3K
        "XME",      # SPDR Metals & Mining - 2K
        "GDXJ",     # VanEck Junior Gold Miners - 3K
        "OIH",      # VanEck Oil Services - 2K
        
        # VOLATILITY - 10K-20K contratos/día
        "VXX",      # iPath Series B S&P 500 VIX - 20K
        "UVXY",     # ProShares Ultra VIX Short-Term - 8K
        "SVXY",     # ProShares Short VIX Short-Term - 6K
        
        # THEMATIC & SPECIALIZED - 5K-15K contratos/día
        "DIA",      # SPDR Dow Jones Industrial Average - 15K
        "BITO",     # ProShares Bitcoin Strategy - 10K
        "ARKK",     # ARK Innovation ETF - 8K
        "JETS",     # U.S. Global Jets - 3K
        "MSOS",     # AdvisorShares Pure US Cannabis - 4K
        "SOXX",     # iShares Semiconductor - 2K
        
        # LEVERAGED SPECIALIZED - 5K-10K contratos/día
        "LABU",     # Direxion Daily S&P Biotech Bull 3X - 10K
        "BOIL",     # ProShares Ultra Bloomberg Natural Gas 2x - 7K
        "TNA",      # Direxion Daily Small Cap Bull 3X - 7K
        "SPXS",     # Direxion Daily S&P 500 Bear -3X - 7K
        "SPXU",     # ProShares UltraPro Short S&P 500 - 6K
        "SOXS",     # Direxion Daily Semiconductor Bear 3X - 5K
        "TZA",      # Direxion Daily Small Cap Bear 3X - 3K
        "TMF",      # Direxion Daily 20+ Year Treasury Bull 3X - 2K
        "TSLL",     # Direxion Daily TSLA Bull 1.5X - 3K
        
        # ADDITIONAL LIQUID ETFs - 2K-5K contratos/día
        "IYR",      # iShares U.S. Real Estate - 3K
        "ASHR",     # Xtrackers Harvest CSI 300 China - 3K
        "UUP",      # Invesco DB US Dollar Index Bullish - 2K
    ]
    
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
