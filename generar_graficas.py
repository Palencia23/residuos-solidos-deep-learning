
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)

# =====================================================
# GRAFICA 1: COMPARACION DE MODELOS
# =====================================================

comparacion = pd.read_csv("resultados/comparacion_modelos.csv")

fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

ejes[0].bar(comparacion["MODELO"], comparacion["MAE"])
ejes[0].set_title("Comparacion del MAE")
ejes[0].set_ylabel("Error absoluto medio (toneladas/dia)")
ejes[0].tick_params(axis="x", rotation=20)

ejes[1].bar(comparacion["MODELO"], comparacion["RMSE"])
ejes[1].set_title("Comparacion del RMSE")
ejes[1].set_ylabel("Raiz del error cuadratico medio")
ejes[1].tick_params(axis="x", rotation=20)

fig.suptitle("Evaluacion de los modelos de prediccion")
fig.tight_layout()

fig.savefig("resultados/grafica_comparacion_modelos.png", dpi=300)
plt.close(fig)

# =====================================================
# GRAFICA 2: SERIE HISTORICA NACIONAL 2010-2024
# =====================================================

historico = pd.read_csv("resultados/serie_nacional_2010_2024.csv")

plt.figure(figsize=(11, 6))

plt.plot(
    historico["AÑO"],
    historico["TONELADAS_PROMEDIO_DIARIO"],
    marker="o"
)

plt.title("Evolucion historica de la disposicion de residuos en Colombia")
plt.xlabel("Año")
plt.ylabel("Toneladas promedio por dia")
plt.xticks(historico["AÑO"], rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("resultados/grafica_serie_historica.png", dpi=300)
plt.close()

# =====================================================
# GRAFICA 3: VALORES REALES VS PREDICCIONES LSTM
# =====================================================

predicciones = pd.read_csv(
    "resultados/predicciones_lstm_mejorado_2024.csv"
)

reales = predicciones["PROMEDIO_REAL"].to_numpy()
estimados = predicciones["PREDICCION_2024"].to_numpy()

# Transformacion log1p para facilitar la visualizacion
reales_log = np.log1p(reales)
estimados_log = np.log1p(estimados)

limite = max(reales_log.max(), estimados_log.max())

plt.figure(figsize=(8, 7))

plt.scatter(reales_log, estimados_log, alpha=0.6)

plt.plot(
    [0, limite],
    [0, limite],
    linestyle="--"
)

plt.title("Valores reales frente a predicciones LSTM mejorado - 2024")
plt.xlabel("Valor real: log(1 + toneladas/dia)")
plt.ylabel("Prediccion: log(1 + toneladas/dia)")
plt.grid(True)
plt.tight_layout()

plt.savefig("resultados/grafica_real_vs_predicho_2024.png", dpi=300)
plt.close()

# =====================================================
# CONFIRMACION
# =====================================================

print("\nGRAFICAS GENERADAS CORRECTAMENTE:")

print("1. resultados/grafica_comparacion_modelos.png")
print("2. resultados/grafica_serie_historica.png")
print("3. resultados/grafica_real_vs_predicho_2024.png")
