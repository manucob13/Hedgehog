# ==============================================================================
# IMPORTS NECESARIOS
# ==============================================================================
import pandas as pd
from typing import Dict, Any

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
    
    Args:
        df_strategy: DataFrame con las piernas de la estrategia
        limit_price: Precio límite de la orden
        host: IP del servidor TWS/Gateway (default: '127.0.0.1')
        port: Puerto de conexión (default: 5000)
        client_id: ID único del cliente (default: 1)
        quantity: Cantidad de contratos por pierna (default: 1)
        tif: Time In Force - 'DAY', 'GTC', etc. (default: 'DAY')
        action: 'BUY' o 'SELL' (default: 'BUY')
        timeout: Timeout de conexión en segundos (default: 10)
    
    Returns:
        Dict con: success, message, trade, order_id, contracts
    """
    
    # ✅ IMPORT LOCAL - Solo cuando se ejecuta la función
    try:
        from ib_insync import IB, Contract, Option, ComboLeg, Order
    except ImportError as e:
        return {
            'success': False,
            'message': f'Error importando ib_insync: {str(e)}. Instala con: pip install ib_insync',
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
        # ================================================================
        # VALIDACIÓN DEL DATAFRAME
        # ================================================================
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
        
        # ================================================================
        # CONEXIÓN A IBKR
        # ================================================================
        print(f"\n🔌 Conectando a IBKR TWS...")
        print(f"   Host: {host}:{port} | Client ID: {client_id}")
        
        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        
        if not ib.isConnected():
            result['message'] = "No se pudo establecer conexión con IBKR"
            print(f"❌ ERROR: {result['message']}")
            return result
        
        print(f"✅ Conexión establecida")
        print(f"   Cuenta(s): {ib.managedAccounts()}")
        
        # ================================================================
        # CREAR CONTRATOS DE OPCIONES
        # ================================================================
        print(f"\n📝 Creando contratos de opciones...")
        contracts = []
        
        for idx, row in df_strategy.iterrows():
            try:
                if row['SecType'].upper() != 'OPT':
                    print(f"⚠️  Fila {idx}: SecType '{row['SecType']}' no es OPT, omitiendo...")
                    continue
                
                # Formatear fecha: YYYY-MM-DD -> YYYYMMDD
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
                if ib and ib.isConnected():
                    ib.disconnect()
                return result
        
        if not contracts:
            result['message'] = "No se crearon contratos válidos"
            print(f"❌ ERROR: {result['message']}")
            if ib and ib.isConnected():
                ib.disconnect()
            return result
        
        # ================================================================
        # CALIFICAR CONTRATOS CON IBKR
        # ================================================================
        print(f"\n🔍 Calificando {len(contracts)} contratos con IBKR...")
        qualified_contracts = ib.qualifyContracts(*contracts)
        
        if len(qualified_contracts) != len(contracts):
            result['message'] = f"Solo se calificaron {len(qualified_contracts)}/{len(contracts)} contratos"
            print(f"⚠️  ADVERTENCIA: {result['message']}")
            
            if len(qualified_contracts) == 0:
                print(f"❌ ERROR: Ningún contrato fue calificado por IBKR")
                if ib and ib.isConnected():
                    ib.disconnect()
                return result
        
        result['contracts'] = qualified_contracts
        
        print(f"\n✅ Contratos calificados exitosamente:")
        for i, c in enumerate(qualified_contracts, 1):
            print(f"   {i}. {c.symbol} {c.lastTradeDateOrContractMonth} "
                  f"{c.strike} {c.right} [conId: {c.conId}]")
        
        # ================================================================
        # CONSTRUIR COMBO BAG (MULTI-LEG)
        # ================================================================
        print(f"\n🎒 Construyendo combo BAG...")
        
        base_symbol = qualified_contracts[0].symbol
        
        combo = Contract()
        combo.symbol = base_symbol
        combo.secType = 'BAG'
        combo.exchange = 'SMART'
        combo.currency = qualified_contracts[0].currency
        combo.comboLegs = []
        
        for idx, row in df_strategy.iterrows():
            # Buscar el contrato calificado correspondiente
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
                if ib and ib.isConnected():
                    ib.disconnect()
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
            
            print(f"   ✓ {leg_action} {leg_quantity}x "
                  f"{matching_contract.symbol} {matching_contract.strike} "
                  f"{matching_contract.right} [conId={matching_contract.conId}]")
        
        print(f"\n✅ Combo BAG creado con {len(combo.comboLegs)} piernas")
        
        # ================================================================
        # CREAR Y ENVIAR ORDEN
        # ================================================================
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
        
        # Esperar para que la orden se procese
        ib.sleep(2)
        
        # Verificar el estado de la orden
        if trade and trade.order:
            result['success'] = True
            result['trade'] = trade
            result['order_id'] = trade.order.orderId
            result['message'] = f"Orden enviada exitosamente - ID: {trade.order.orderId}"
            
            print(f"\n✅ ¡ORDEN ENVIADA EXITOSAMENTE!")
            print(f"   Order ID: {trade.order.orderId}")
            
            if hasattr(trade, 'orderStatus') and trade.orderStatus:
                print(f"   Estado: {trade.orderStatus.status}")
            else:
                print(f"   Estado: Pendiente de confirmación")
            
        else:
            result['message'] = "Orden enviada pero no se pudo verificar el estado"
            print(f"\n⚠️  {result['message']}")
        
    except Exception as e:
        result['success'] = False
        result['message'] = f"Error durante la ejecución: {str(e)}"
        print(f"\n❌ ERROR CRÍTICO: {result['message']}")
        
        # Mostrar traceback completo para debugging
        import traceback
        print("\n📋 TRACEBACK COMPLETO:")
        print(traceback.format_exc())
        
    finally:
        # Siempre desconectar al finalizar
        if ib and ib.isConnected():
            print(f"\n🔌 Cerrando conexión con IBKR...")
            ib.disconnect()
            print(f"✅ Conexión cerrada correctamente")
        
        print(f"\n{'='*60}")
        print(f"🏁 PROCESO FINALIZADO")
        print(f"{'='*60}\n")
    
    return result


# ==============================================================================
# FUNCIÓN DE PRUEBA DE CONEXIÓN
# ==============================================================================

def test_ibkr_connection(
    host: str = '127.0.0.1',
    port: int = 5000,
    client_id: int = 999
) -> Dict[str, Any]:
    """
    Prueba la conexión con IBKR TWS/Gateway.
    
    Args:
        host: IP del servidor (default: '127.0.0.1')
        port: Puerto de conexión (default: 5000)
        client_id: ID único del cliente (default: 999)
    
    Returns:
        Dict con: success, message, accounts
    """
    try:
        from ib_insync import IB
    except ImportError as e:
        return {
            'success': False,
            'message': f'Error importando ib_insync: {str(e)}',
            'accounts': []
        }
    
    result = {
        'success': False,
        'message': '',
        'accounts': []
    }
    
    ib = None
    
    try:
        print(f"\n{'='*60}")
        print(f"🧪 PRUEBA DE CONEXIÓN IBKR")
        print(f"{'='*60}")
        print(f"🔌 Intentando conectar a {host}:{port} (Client ID: {client_id})...")
        
        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=10)
        
        if ib.isConnected():
            accounts = ib.managedAccounts()
            result['success'] = True
            result['message'] = f"Conexión exitosa a {host}:{port}"
            result['accounts'] = accounts
            
            print(f"✅ ¡CONEXIÓN EXITOSA!")
            print(f"   Host: {host}:{port}")
            print(f"   Client ID: {client_id}")
            print(f"   Cuentas: {accounts}")
            
        else:
            result['message'] = "No se pudo establecer conexión"
            print(f"❌ {result['message']}")
        
    except Exception as e:
        result['message'] = f"Error de conexión: {str(e)}"
        print(f"❌ ERROR: {result['message']}")
        
        import traceback
        print("\n📋 TRACEBACK:")
        print(traceback.format_exc())
        
    finally:
        if ib and ib.isConnected():
            ib.disconnect()
            print(f"\n🔌 Conexión cerrada")
        
        print(f"\n{'='*60}")
        print(f"🏁 PRUEBA FINALIZADA")
        print(f"{'='*60}\n")
    
    return result


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("EJEMPLO DE USO - send_strategy_order_ibkr")
    print("="*60)
    
    # 1. Primero, probar la conexión
    print("\n1️⃣ Probando conexión...")
    connection_result = test_ibkr_connection(
        host='127.0.0.1',
        port=5000,
        client_id=999
    )
    
    if not connection_result['success']:
        print("\n❌ No se pudo conectar a IBKR. Verifica:")
        print("   - TWS/Gateway está ejecutándose")
        print("   - El puerto es correcto (5000 en tu caso)")
        print("   - La API está habilitada en TWS")
        print("   - No hay otra conexión con el mismo Client ID")
        exit(1)
    
    print("\n✅ Conexión exitosa! Procediendo con ejemplo de orden...")
    
    # 2. Crear un DataFrame de ejemplo - Triple Calendar
    df_example = pd.DataFrame([
        # DOWN Leg
        {'Action': 'SELL', 'Quantity': 1, 'Symbol': 'QQQ', 'SecType': 'OPT', 
         'Expiry': '2025-01-24', 'Strike': 500, 'Right': 'P', 
         'Exchange': 'SMART', 'Currency': 'USD'},
        {'Action': 'BUY', 'Quantity': 1, 'Symbol': 'QQQ', 'SecType': 'OPT', 
         'Expiry': '2025-01-31', 'Strike': 500, 'Right': 'P', 
         'Exchange': 'SMART', 'Currency': 'USD'},
        
        # ATM Leg
        {'Action': 'SELL', 'Quantity': 1, 'Symbol': 'QQQ', 'SecType': 'OPT', 
         'Expiry': '2025-01-24', 'Strike': 510, 'Right': 'P', 
         'Exchange': 'SMART', 'Currency': 'USD'},
        {'Action': 'BUY', 'Quantity': 1, 'Symbol': 'QQQ', 'SecType': 'OPT', 
         'Expiry': '2025-01-31', 'Strike': 510, 'Right': 'P', 
         'Exchange': 'SMART', 'Currency': 'USD'},
        
        # UP Leg
        {'Action': 'SELL', 'Quantity': 1, 'Symbol': 'QQQ', 'SecType': 'OPT', 
         'Expiry': '2025-01-24', 'Strike': 520, 'Right': 'C', 
         'Exchange': 'SMART', 'Currency': 'USD'},
        {'Action': 'BUY', 'Quantity': 1, 'Symbol': 'QQQ', 'SecType': 'OPT', 
         'Expiry': '2025-01-31', 'Strike': 520, 'Right': 'C', 
         'Exchange': 'SMART', 'Currency': 'USD'},
    ])
    
    print("\n2️⃣ DataFrame de estrategia:")
    print(df_example.to_string(index=False))
    
    # 3. Enviar la orden (DESCOMENTA PARA ENVIAR ORDEN REAL)
    """
    print("\n3️⃣ Enviando orden a IBKR...")
    
    result = send_strategy_order_ibkr(
        df_strategy=df_example,
        limit_price=0.50,  # Precio límite de la estrategia
        host='127.0.0.1',
        port=5000,
        client_id=1,
        quantity=1,
        tif='DAY',
        action='BUY',
        timeout=10
    )
    
    if result['success']:
        print(f"\n🎉 ¡ÉXITO! Orden enviada")
        print(f"   Order ID: {result['order_id']}")
        print(f"   Contratos calificados: {len(result['contracts'])}")
    else:
        print(f"\n❌ Error: {result['message']}")
    """
    
    print("\n⚠️  Para enviar una orden real, descomenta el bloque de código anterior")
    print("="*60)
