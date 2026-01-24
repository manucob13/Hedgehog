    # ===== GRÁFICO 6: DOCUMENTACIÓN DE LÓGICA =====
    ax6 = fig.add_subplot(gs[5])
    ax6.axis('off')
    
    # Tabla de documentación (más compacta y legible)
    doc_text = """
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║         LÓGICA DE CLASIFICACIÓN - Z-Score MACD-V Analyzer v3.0 (7 Estados)                                                                    ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ VARIABLES:                                                                                                                                     ║
║   • Z-Score Adjusted: momentum MACD-V normalizado, ajustado por curtosis (-4σ a +4σ)                                                          ║
║   • MACD-V: MACD / ATR × 100. MACD-V > Signal = momentum alcista                                                                               ║
║   • SMA(50): precio > SMA50 = contexto alcista, precio < SMA50 = contexto bajista                                                             ║
║   • ΔMACD-V: dirección del MACD-V (MACD_V[t] - MACD_V[t-1]) → separa Fuerte (acelera) vs Débil (desacelera)                                   ║
║                                                                                                                                                ║
║ ESTADOS ALCISTAS (Precio > SMA50):                                                                                                            ║
║   🔴 SOBRECOMPRA_FUERTE: Z > 2.0σ + MACD-V > Signal + ΔMACD-V > 0                                                                              ║
║      → Extremo alcista acelerando. Máximo riesgo pero momentum fuerte.                                                                         ║
║   🔴 SOBRECOMPRA_DEBIL: Z > 2.0σ + MACD-V > Signal + ΔMACD-V ≤ 0                                                                               ║
║      → Extremo alcista pero momentum se debilita. Señal de corrección próxima.                                                                 ║
║   🔵 ALCISTA: 0.75σ < Z ≤ 2.0σ + MACD-V > Signal                                                                                               ║
║      → Tendencia alcista confirmada. Mejor zona para entrada.                                                                                  ║
║                                                                                                                                                ║
║ ESTADOS BAJISTAS (Precio < SMA50):                                                                                                            ║
║   🟣 SOBREVENTA_FUERTE: Z < -2.0σ + MACD-V < Signal + ΔMACD-V < 0                                                                              ║
║      → Extremo bajista acelerando. Máxima presión vendedora.                                                                                   ║
║   🟣 SOBREVENTA_DEBIL: Z < -2.0σ + MACD-V < Signal + ΔMACD-V ≥ 0                                                                               ║
║      → Extremo bajista pero momentum se debilita. Posible rebote.                                                                              ║
║   🔴 BAJISTA: -2.0σ ≤ Z < -0.75σ + MACD-V < Signal                                                                                             ║
║      → Tendencia bajista confirmada. Evitar compra.                                                                                            ║
║                                                                                                                                                ║
║ ZONA NEUTRAL:                                                                                                                                  ║
║   🟡 RANGO: -0.75σ ≤ Z ≤ 0.75σ (independiente de SMA50 y MACD-V)                                                                               ║
║      → Consolidación sin dirección clara. Esperar ruptura.                                                                                     ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
v3.0: Añade separación Fuerte/Débil usando ΔMACD-V para detectar si extremos se están extendiendo o agotando.
    """
    
    ax6.text(0.5, 0.5, doc_text, transform=ax6.transAxes, fontsize=9, verticalalignment='center',
            horizontalalignment='center', family='monospace', 
            bbox=dict(boxstyle='round', facecolor='#1A1D29', alpha=0.98, edgecolor='#00D9FF', linewidth=2.5),
            color='#FFFFFF', linespacing=1.4)
    
    plt.tight_layout()
    return fig
