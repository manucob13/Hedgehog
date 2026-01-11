# ==============================================================================
# IMPORTS NECESARIOS
# ==============================================================================
import pandas as pd
from typing import Dict, Any
import asyncio
import sys

# ==============================================================================
# CONFIGURACIÓN DE EVENT LOOP PARA STREAMLIT CLOUD
# ==============================================================================
def setup_event_loop():
    """
    Configura el event loop de asyncio para que funcione en Streamlit Cloud.
    Necesario para ib_insync.
    """
    try:
        # Intentar obtener el event loop actual
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Si no hay event loop o está cerrado, crear uno nuevo
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# ==============================================================================
# FUNCIONES IBKR
# ==============================================================================

def send_strategy_order_ibkr(
    df_strategy: pd.DataFrame,
    limit_price: float,
    host: str = '127.0.0.1',
    port: int = 5000,
    client_id: int = 1,
    quantity: int = 1,
    tif: str = 'DAY',
    action: str = 'BUY',
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Envía una orden de estrategia multi-leg a IBKR TWS/Gateway.
    
    ⚠️ IMPORTANTE: Esta función importa ib_insync localmente para evitar
    problemas con event loops en Streamlit Cloud.
    """
    
    # ✅ CONFIGURAR EVENT LOOP ANTES DE IMPORTAR ib_insync
    setup_event_loop()
    
    # ✅ IMPORT LOCAL - Solo cuando se ejecuta la función
    try:
        from ib_insync import IB, Contract, Option, ComboLeg, Order, util
    except ImportError as e:
        return {
            'success': False,
            'message': f'Error importando ib_insync: {str(e)}',
            'trade': None,
            'order_id': None,
            'contracts': []
        }
    
    ib = None
    result = {
        'success': False,
        'message': '',
        'trade': None,
        'order_id': None,
        'contracts': []
    }
    
    try:
        # Validación del DataFrame
        required_columns = ['Action', 'Quantity', 'Symbol', 'SecType', 'Expiry', 
                          'Strike', 'Right', 'Exchange', 'Currency']
        missing_cols = [col for col in required_columns if col not in df_strategy.columns]
        
        if missing_cols:
            result['message'] = f"Columnas faltantes en DataFrame: {missing_cols}"
            print(f"❌ ERROR: {result['message']}")
            return result
        
        if df_strategy.empty:
            result['message'] = "DataFrame vacío - no hay piernas para procesar"
            print(f"❌ ERROR: {result['message']}")
            return result
        
        print(f"\n{'='*60}")
        print(f"🔄 INICIANDO ENVÍO DE ESTRATEGIA A IBKR")
        print(f"{'='*60}")
        print(f"📊 Piernas a procesar: {len(df_strategy)}")
        print(f"💰 Precio límite: ${limit_price:.2f}")
        print(f"📦 Cantidad: {quantity}")
        print(f"🎯 Acción: {action}")
        
        # Conexión a IBKR
        print(f"\n🔌 Conectando a IBKR TWS...")
        print(f"   Host: {host}:{port} | Client ID: {client_id}")
        
        util.startLoop()
        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        
        if not ib.isConnected():
            result['message'] = "No se pudo establecer conexión con IBKR"
            print(f"❌ ERROR: {result['message']}")
            return result
        
        print(f"✅ Conexión establecida: {ib}")
        
        # Crear y calificar contratos
        print(f"\n📝 Creando contratos de opciones...")
        contracts = []
        
        for idx, row in df_strategy.iterrows():
            try:
                if row['SecType'].upper() != 'OPT':
                    print(f"⚠️  Fila {idx}: SecType '{row['SecType']}' no es OPT, omitiendo...")
                    continue
                
                expiry = str(row['Expiry']).replace('-', '')
                
                contract = Option(
                    symbol=str(row['Symbol']),
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(row['Strike']),
                    right=str(row['Right']).upper(),
                    exchange=str(row['Exchange']),
                    currency=str(row['Currency'])
                )
                
                contracts.append(contract)
                print(f"   ✓ {row['Symbol']} {expiry} {row['Strike']} {row['Right']}")
                
            except Exception as e:
                result['message'] = f"Error creando contrato en fila {idx}: {str(e)}"
                print(f"❌ {result['message']}")
                return result
        
        if not contracts:
            result['message'] = "No se crearon contratos válidos"
            print(f"❌ ERROR: {result['message']}")
            return result
        
        print(f"\n🔍 Calificando {len(contracts)} contratos con IBKR...")
        qualified_contracts = ib.qualifyContracts(*contracts)
        
        if len(qualified_contracts) != len(contracts):
            result['message'] = f"Solo se calificaron {len(qualified_contracts)}/{len(contracts)} contratos"
            print(f"⚠️  ADVERTENCIA: {result['message']}")
        
        result['contracts'] = qualified_contracts
        
        print(f"\n✅ Contratos calificados:")
        for c in qualified_contracts:
            print(f"   • {c.symbol} {c.lastTradeDateOrContractMonth} {c.strike} {c.right} [conId: {c.conId}]")
        
        # Construir combo BAG
        print(f"\n🎒 Construyendo combo BAG...")
        
        base_symbol = qualified_contracts[0].symbol
        
        combo = Contract()
        combo.symbol = base_symbol
        combo.secType = 'BAG'
        combo.exchange = 'SMART'
        combo.currency = qualified_contracts[0].currency
        combo.comboLegs = []
        
        for idx, row in df_strategy.iterrows():
            matching_contract = None
            expiry = str(row['Expiry']).replace('-', '')
            
            for c in qualified_contracts:
                if (c.symbol == row['Symbol'] and 
                    c.lastTradeDateOrContractMonth == expiry and
                    c.strike == float(row['Strike']) and
                    c.right == str(row['Right']).upper()):
                    matching_contract = c
                    break
            
            if not matching_contract:
                result['message'] = f"No se encontró contrato calificado para fila {idx}"
                print(f"❌ ERROR: {result['message']}")
                return result
            
            leg_action = str(row['Action']).upper()
            leg_quantity = int(row['Quantity'])
            
            combo_leg = ComboLeg(
                conId=matching_contract.conId,
                ratio=leg_quantity,
                action=leg_action,
                exchange=str(row['Exchange'])
            )
            combo.comboLegs.append(combo_leg)
            
            print(f"   ✓ {leg_action} {leg_quantity}x conId={matching_contract.conId}")
        
        print(f"\n✅ Combo creado con {len(combo.comboLegs)} piernas")
        
        # Crear y enviar orden
        print(f"\n📤 Preparando orden LIMIT...")
        
        order = Order(
            action=action,
            orderType='LMT',
            totalQuantity=quantity,
            lmtPrice=round(limit_price, 2),
            transmit=True,
            tif=tif
        )
        
        print(f"\n{'='*60}")
        print(f"📋 RESUMEN DE ORDEN")
        print(f"{'='*60}")
        print(f"   Estrategia: {base_symbol} BAG ({len(combo.comboLegs)} piernas)")
        print(f"   Acción: {action}")
        print(f"   Cantidad: {quantity}")
        print(f"   Tipo: LIMIT @ ${limit_price:.2f}")
        print(f"   TIF: {tif}")
        print(f"{'='*60}\n")
        
        print(f"🚀 Enviando orden a IBKR...")
        trade = ib.placeOrder(combo, order)
        
        ib.sleep(2)
        ib.reqOpenOrders()
        ib.sleep(1)
        
        if trade and trade.order:
            result['success'] = True
            result['trade'] = trade
            result['order_id'] = trade.order.orderId
            result['message'] = f"Orden enviada exitosamente - ID: {trade.order.orderId}"
            
            print(f"\n✅ ¡ORDEN ENVIADA EXITOSAMENTE!")
            print(f"   Order ID: {trade.order.orderId}")
            print(f"   Estado: {trade.orderStatus.status}")
            
        else:
            result['message'] = "Orden enviada pero no se pudo verificar el estado"
            print(f"\n⚠️  {result['message']}")
        
    except Exception as e:
        result['success'] = False
        result['message'] = f"Error durante la ejecución: {str(e)}"
        print(f"\n❌ ERROR CRÍTICO: {result['message']}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        if ib and ib.isConnected():
            print(f"\n🔌 Cerrando conexión con IBKR...")
            ib.disconnect()
            print(f"✅ Conexión cerrada correctamente")
        
        print(f"\n{'='*60}")
        print(f"🏁 PROCESO FINALIZADO")
        print(f"{'='*60}\n")
    
    return result
