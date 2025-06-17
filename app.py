import streamlit as st
import pandas as pd
import joblib

# Título
st.title("Predicción de Enfermedad Renal Crónica (CKD)")
st.write("Esta aplicación usa un modelo de Árbol de Decisión para predecir si un paciente tiene enfermedad renal crónica.")

# Selección del modelo
model_option = st.selectbox("Selecciona el modelo:", ["decision_tree_model.pkl", "best_decision_tree_model.pkl"])

# Carga del modelo
#@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_option)

# Lista de columnas esperadas por el modelo
input_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad',
                  'appet', 'pe', 'ane']

# Diccionarios para opciones categóricas
categorical_options = {
    'rbc': ['normal', 'abnormal'],
    'pc': ['normal', 'abnormal'],
    'pcc': ['present', 'notpresent'],
    'ba': ['present', 'notpresent'],
    'htn': ['yes', 'no'],
    'dm': ['yes', 'no'],
    'cad': ['yes', 'no'],
    'appet': ['good', 'poor'],
    'pe': ['yes', 'no'],
    'ane': ['yes', 'no']
}

# Crear un formulario para ingresar los valores
with st.form("input_form"):
    st.subheader("Ingrese los datos del paciente")
    user_input = {}
    for feature in input_features:
        if feature in categorical_options:
            user_input[feature] = st.selectbox(f"{feature}", categorical_options[feature])
        else:
            user_input[feature] = st.number_input(f"{feature}", step=0.1)

    submitted = st.form_submit_button("Predecir")

# Realizar la predicción
if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    # Mapear salida
    resultado = "ckd (Tiene enfermedad renal crónica)" if prediction == 0 else "notckd (NO tiene enfermedad renal)"
    st.success(f"Predicción del modelo: **{resultado}**")
