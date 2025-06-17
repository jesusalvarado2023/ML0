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

# Columnas esperadas
input_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad',
                  'appet', 'pe', 'ane']

# Opciones categóricas y su codificación
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

# Mapeo inverso (para valores por defecto)
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

# Valores por defecto
default_values = [40, 2, 0, 4, 0, 1, 0, 1, 0, 44, 43, 32, 2, 0, 48, 19, 25, 17, 1, 1, 0, 1, 1, 1]

# Diccionario con valores por defecto legibles
default_dict = {}
for i, feature in enumerate(input_features):
    if feature in categorical_options:
        default_dict[feature] = category_mapping[feature][default_values[i]]
    else:
        default_dict[feature] = default_values[i]

# Formulario de entrada
with st.form("input_form"):
    st.subheader("Ingrese los datos del paciente (o use los valores por defecto)")
    user_input = {}
    for feature in input_features:
        if feature in categorical_options:
            options = categorical_options[feature]
            user_input[feature] = st.selectbox(f"{feature}", options, index=options.index(default_dict[feature]))
        else:
            user_input[feature] = st.number_input(f"{feature}", value=float(default_dict[feature]), step=0.1)
    submitted = st.form_submit_button("Predecir")

if submitted:
    # Mostrar la tabla original ingresada
    input_df_display = pd.DataFrame([user_input])
    st.write("### Datos ingresados:")
    st.dataframe(input_df_display)

    # Codificar las variables categóricas como números (Label Encoding)
    input_df_encoded = input_df_display.copy()
    for feature in categorical_options:
        input_df_encoded[feature] = input_df_encoded[feature].apply(lambda x: categorical_options[feature].index(x))

    # Mostrar la tabla codificada
    st.write("### Datos procesados para el modelo (codificados):")
    st.dataframe(input_df_encoded)

    # Predecir
    try:
        prediction = model.predict(input_df_encoded)[0]
        resultado = "ckd (Tiene enfermedad renal crónica)" if prediction == 0 else "notckd (NO tiene enfermedad renal)"
        st.success(f"Predicción del modelo: **{resultado}**")
    except Exception as e:
        st.error(f"Error al predecir: {e}")
