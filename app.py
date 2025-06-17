import streamlit as st
import pandas as pd
import joblib

st.title("Predicción de Enfermedad Renal Crónica (CKD)")

# Selección del modelo
model_option = st.selectbox("Selecciona el modelo:", ["decision_tree_model.pkl", "best_decision_tree_model.pkl"])

#@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_option)

# Columnas esperadas por el modelo
input_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad',
                  'appet', 'pe', 'ane']

# Mapeo de variables categóricas
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

# Valores por defecto en la fila de datos
default_values = [40, 2, 0, 4, 0, 1, 0, 1, 0, 44, 43, 32, 2, 0, 48, 19, 25, 17, 1, 1, 0, 1, 1, 1]

# Mapeo para las categorías
category_mapping = {
    'rbc': {0: 'normal', 1: 'abnormal'},
    'pc': {0: 'normal', 1: 'abnormal'},
    'pcc': {0: 'present', 1: 'notpresent'},
    'ba': {0: 'present', 1: 'notpresent'},
    'htn': {0: 'no', 1: 'yes'},
    'dm': {0: 'no', 1: 'yes'},
    'cad': {0: 'no', 1: 'yes'},
    'appet': {0: 'good', 1: 'poor'},
    'pe': {0: 'no', 1: 'yes'},
    'ane': {0: 'no', 1: 'yes'}
}

# Convertir default values en diccionario para mostrar por defecto
default_dict = {}
for i, feature in enumerate(input_features):
    if feature in categorical_options:
        default_dict[feature] = category_mapping[feature][default_values[i]]
    else:
        default_dict[feature] = default_values[i]

# Formulario
with st.form("input_form"):
    st.subheader("Ingrese los datos del paciente (o use los valores por defecto)")
    user_input = {}
    for feature in input_features:
        if feature in categorical_options:
            user_input[feature] = st.selectbox(f"{feature}", categorical_options[feature], index=categorical_options[feature].index(default_dict[feature]))
        else:
            user_input[feature] = st.number_input(f"{feature}", value=float(default_dict[feature]), step=0.1)

    submitted = st.form_submit_button("Predecir")

# Predicción
if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    resultado = "ckd (Tiene enfermedad renal crónica)" if prediction == 0 else "notckd (NO tiene enfermedad renal)"
    st.success(f"Predicción del modelo: **{resultado}**")
