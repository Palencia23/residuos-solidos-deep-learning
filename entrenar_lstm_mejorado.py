
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# 1. CONFIGURACION
# =====================================================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

ARCHIVO = "resultados/panel_municipal_completo_lstm.csv"
CARPETA = "resultados"

os.makedirs(CARPETA, exist_ok=True)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_csv(ARCHIVO)

codigo = "COD DANE ATENDIDO"
anio = "AÑO"
valor = "PROMEDIO T/DIA"
municipio = "MUNICIPIO"

df[codigo] = df[codigo].astype(str).str.strip()

tabla = df.pivot_table(
    index=codigo,
    columns=anio,
    values=valor,
    aggfunc="mean"
).sort_index()

nombres = (
    df.drop_duplicates(codigo)
    .set_index(codigo)[municipio]
)

tabla = tabla.loc[:, list(range(2010, 2025))]
tabla = tabla.dropna()

codigos = tabla.index.tolist()
valores = tabla.values.astype("float32")

print("MUNICIPIOS:", len(codigos))
print("AÑOS:", tabla.columns.min(), "-", tabla.columns.max())

# =====================================================
# 3. TRANSFORMACION Y ESCALADO
# =====================================================

valores_log = np.log1p(valores)

escalador = MinMaxScaler()

escalador.fit(valores_log[:, :12].reshape(-1, 1))

valores_escalados = escalador.transform(
    valores_log.reshape(-1, 1)
).reshape(valores_log.shape)

# =====================================================
# 4. CREAR SECUENCIAS
# =====================================================

X_train = []
ID_train = []
y_train = []

X_val = []
ID_val = []
y_val = []

X_test = []
ID_test = []
y_test = []

for i in range(len(codigos)):

    for j in range(5, 15):

        secuencia = valores_escalados[i, j-5:j]
        objetivo = valores_escalados[i, j]

        if j <= 11:
            X_train.append(secuencia)
            ID_train.append(i)
            y_train.append(objetivo)

        elif j <= 13:
            X_val.append(secuencia)
            ID_val.append(i)
            y_val.append(objetivo)

        else:
            X_test.append(secuencia)
            ID_test.append(i)
            y_test.append(objetivo)

X_train = np.array(X_train).reshape(-1, 5, 1)
X_val = np.array(X_val).reshape(-1, 5, 1)
X_test = np.array(X_test).reshape(-1, 5, 1)

ID_train = np.array(ID_train)
ID_val = np.array(ID_val)
ID_test = np.array(ID_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print("\nMUESTRAS DE ENTRENAMIENTO:", len(X_train))
print("MUESTRAS DE VALIDACION:", len(X_val))
print("MUESTRAS DE PRUEBA:", len(X_test))

# =====================================================
# 5. CONSTRUIR LSTM CON IDENTIFICADOR MUNICIPAL
# =====================================================

entrada_historial = tf.keras.Input(shape=(5, 1), name="historial")

entrada_municipio = tf.keras.Input(shape=(1,), dtype="int32",
                                   name="municipio")

lstm = tf.keras.layers.LSTM(32)(entrada_historial)

embedding = tf.keras.layers.Embedding(
    input_dim=len(codigos),
    output_dim=16
)(entrada_municipio)

embedding = tf.keras.layers.Flatten()(embedding)

combinacion = tf.keras.layers.Concatenate()([lstm, embedding])

capa = tf.keras.layers.Dense(32, activation="relu")(combinacion)

salida = tf.keras.layers.Dense(1, activation="relu")(capa)

modelo = tf.keras.Model(
    inputs=[entrada_historial, entrada_municipio],
    outputs=salida
)

modelo.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

modelo.summary()

# =====================================================
# 6. ENTRENAMIENTO
# =====================================================

detener = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

historial = modelo.fit(
    [X_train, ID_train],
    y_train,
    validation_data=([X_val, ID_val], y_val),
    epochs=100,
    batch_size=64,
    callbacks=[detener],
    verbose=1
)

# =====================================================
# 7. PREDICCIONES 2024
# =====================================================

predicciones_escaladas = modelo.predict(
    [X_test, ID_test],
    verbose=0
)

predicciones_log = escalador.inverse_transform(
    predicciones_escaladas
).flatten()

predicciones = np.expm1(predicciones_log)

reales = valores[:, 14]

resultado = pd.DataFrame({
    "COD DANE ATENDIDO": codigos,
    "MUNICIPIO": [nombres.loc[c] for c in codigos],
    "PROMEDIO_REAL": reales,
    "PREDICCION_2024": predicciones
})

resultado["ERROR_ABSOLUTO"] = (
    resultado["PROMEDIO_REAL"] -
    resultado["PREDICCION_2024"]
).abs()

resultado.to_csv(
    f"{CARPETA}/predicciones_lstm_mejorado_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 8. EVALUACION
# =====================================================

mae = mean_absolute_error(reales, predicciones)

rmse = np.sqrt(
    mean_squared_error(reales, predicciones)
)

r2 = r2_score(reales, predicciones)

# Referencia: repetir el valor de 2023

referencia = valores[:, 13]

mae_ref = mean_absolute_error(reales, referencia)
rmse_ref = np.sqrt(mean_squared_error(reales, referencia))
r2_ref = r2_score(reales, referencia)

print("\n========== LSTM MEJORADO ==========")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

print("\n========== REFERENCIA 2023 ==========")
print("MAE:", mae_ref)
print("RMSE:", rmse_ref)
print("R2:", r2_ref)

print("\n========== COMPARACION ==========")

if mae < mae_ref and rmse < rmse_ref:
    print("EL LSTM MEJORADO SUPERA LA REFERENCIA EN MAE Y RMSE.")
else:
    print("EL LSTM MEJORADO TODAVIA NO SUPERA LA REFERENCIA EN AMBAS METRICAS.")

modelo.save(f"{CARPETA}/modelo_lstm_mejorado.keras")

print("\nARCHIVOS GUARDADOS CORRECTAMENTE.")
