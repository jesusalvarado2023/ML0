import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar modelo
# @st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("modelo/best_decision_tree_model.pkl")

model = load_model()

st.title("Predicción con Árbol de Decisión")
st.subheader("Introduce los datos del paciente")

# Lista de campos esperados por el modelo
features = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm",
    "cad", "appet", "pe", "ane"
]

# Valores posibles para algunas variables categóricas (modifícalo según tu dataset)
binary_map = {
    0: "No",
    1: "Sí"
}

form = st.form(key="input_form")

# Crear inputs dinámicamente
inputs = {}
for feature in features:
    if feature in ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]:
        value = form.selectbox(f"{feature}", options=[0, 1], format_func=lambda x: binary_map[x], key=feature)
    else:
        value = form.number_input(f"{feature}", key=feature)
    inputs[feature] = value

submit = form.form_submit_button("Predecir")

if submit:
    # Convertir en DataFrame con una sola fila
    input_df = pd.DataFrame([inputs])
    
    st.write("### Datos ingresados:")
    st.dataframe(input_df)

    # Realizar predicción
    prediction = model.predict(input_df)[0]
    
    st.write("### Predicción del modelo:")
    st.success(f"Resultado: {prediction}")
