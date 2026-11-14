
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# 1. Cargar datos
archivo = "resultados/panel_municipal_completo_lstm.csv"
df = pd.read_csv(archivo, dtype={"COD DANE ATENDIDO": str})

anios = list(range(2010, 2025))

# Organizar municipios por años
panel = df.pivot(
    index="COD DANE ATENDIDO",
    columns="AÑO",
    values="PROMEDIO T/DIA"
).reindex(columns=anios)

print("Municipios:", len(panel))
print("Años:", len(anios))

# 2. Transformación logarítmica
valores = panel.values.astype(float)
valores_log = np.log1p(valores)

# 3. Normalización: ajustar solo con datos hasta 2021
scaler = MinMaxScaler()
scaler.fit(valores_log[:, :12].reshape(-1, 1))

valores_escalados = scaler.transform(
    valores_log.reshape(-1, 1)
).reshape(valores_log.shape)

# 4. Crear secuencias de cinco años
def crear_secuencias(anios_objetivo):
    X, y = [], []

    for anio in anios_objetivo:
        indice = anios.index(anio)

        entradas = valores_escalados[
            :, indice - 5:indice
        ]

        objetivos = valores_escalados[:, indice]

        X.append(entradas)
        y.append(objetivos)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X.reshape(-1, 5, 1), y

X_train, y_train = crear_secuencias(
    range(2015, 2022)
)

X_val, y_val = crear_secuencias(
    [2022, 2023]
)

X_test, y_test = crear_secuencias(
    [2024]
)

print("\n========== SECUENCIAS ==========")
print("Entrenamiento:", X_train.shape, y_train.shape)
print("Validación:", X_val.shape, y_val.shape)
print("Prueba:", X_test.shape, y_test.shape)

# 5. Construir la red neuronal LSTM
modelo = Sequential([
    Input(shape=(5, 1)),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1, activation="relu")
])

modelo.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

modelo.summary()

# 6. Entrenar
detener = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

historial = modelo.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[detener],
    verbose=1
)

# 7. Evaluación final en 2024
predicciones_escaladas = modelo.predict(
    X_test, verbose=0
)

predicciones_log = scaler.inverse_transform(
    predicciones_escaladas
).flatten()

reales_log = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predicciones = np.maximum(
    np.expm1(predicciones_log), 0
)

reales = np.expm1(reales_log)

mae = mean_absolute_error(reales, predicciones)
rmse = np.sqrt(mean_squared_error(reales, predicciones))
r2 = r2_score(reales, predicciones)

print("\n========== RESULTADOS 2024 ==========")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

# 8. Guardar predicciones
municipios = df[
    ["COD DANE ATENDIDO", "DEPARTAMENTO", "MUNICIPIO"]
].drop_duplicates("COD DANE ATENDIDO")

resultado = municipios.copy()
resultado["REAL_2024"] = reales
resultado["PREDICCION_2024"] = predicciones

resultado.to_csv(
    "resultados/predicciones_lstm_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

# 9. Guardar modelo
modelo.save("resultados/modelo_lstm_residuos.keras")

print("\nArchivos guardados:")
print("resultados/predicciones_lstm_2024.csv")
print("resultados/modelo_lstm_residuos.keras")
 