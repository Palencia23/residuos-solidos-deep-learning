
import pandas as pd
import os

os.makedirs("resultados", exist_ok=True)

comparacion = pd.DataFrame({
    "MODELO": [
        "LSTM inicial",
        "LSTM mejorado con Embedding",
        "Referencia: repetir valor de 2023"
    ],
    "MAE": [
        5.89578557936053,
        4.410475730895996,
        2.8164455890655518
    ],
    "RMSE": [
        39.259445038422406,
        19.19095503425906,
        15.348395427128532
    ],
    "R2": [
        0.9762853415463724,
        0.9943333864212036,
        0.9963754415512085
    ]
})

print("\n========== COMPARACION FINAL DE MODELOS ==========\n")
print(comparacion.to_string(index=False))

comparacion.to_csv(
    "resultados/comparacion_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo guardado: resultados/comparacion_modelos.csv")

print("\n========== MEJORA DEL SEGUNDO LSTM ==========")

mae_inicial = comparacion.loc[0, "MAE"]
mae_mejorado = comparacion.loc[1, "MAE"]

rmse_inicial = comparacion.loc[0, "RMSE"]
rmse_mejorado = comparacion.loc[1, "RMSE"]

print(
    "Reduccion MAE:",
    round((mae_inicial - mae_mejorado) / mae_inicial * 100, 2),
    "%"
)

print(
    "Reduccion RMSE:",
    round((rmse_inicial - rmse_mejorado) / rmse_inicial * 100, 2),
    "%"
)
