import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
#@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("best_decision_tree_model.pkl")

model = load_model()

# Interfaz
st.title("Predicción con Árbol de Decisión")

st.write("Sube un archivo CSV con las características de entrada para predecir.")

uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(data)

    # Realizar predicciones
    predictions = model.predict(data)
    st.write("### Predicciones:")
    st.write(predictions)

    # Adjuntar predicciones al DataFrame original
    result_df = data.copy()
    result_df["Predicción"] = predictions

    # Mostrar resultados
    st.dataframe(result_df)

    # Permitir descargar los resultados
    csv = result_df.to_csv(index=False)
    st.download_button("Descargar resultados como CSV", csv, "predicciones.csv", "text/csv")
