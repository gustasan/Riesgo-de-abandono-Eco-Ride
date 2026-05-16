
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Rutas de los archivos guardados (ajusta si es necesario para tu entorno de despliegue)
# En este ejemplo, asumimos que app.py estará en la misma carpeta que los .pkl
model_path = 'modelo churn.pkl'
preprocessor_path = 'pipeline preproc.pkl'

# Cargar el modelo y el preprocesador
@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_resources()

# Título de la aplicación
st.title('Sistema de Alerta Temprana de Churn - Eco-Ride')
st.markdown("--- ")

# Controles web interactivos para la entrada de datos del cliente
st.header('Ingrese los Datos del Cliente:')

edad = st.slider('Edad', min_value=18, max_value=100, value=30)
plan_options = ['Básico', 'Premium', 'Elite']
plan = st.selectbox('Plan', options=plan_options)
uso_mensual_km = st.slider('Uso Mensual Km', min_value=0.0, max_value=200.0, value=75.0, step=0.1)
soporte_tickets = st.slider('Soporte Tickets', min_value=0, max_value=10, value=1)
gasto_promedio = st.slider('Gasto Promedio', min_value=10.0, max_value=5000.0, value=75.0, step=0.1)
region_options = ['Norte', 'Sur', 'Centro']
region = st.selectbox('Región', options=region_options)

# Botón para analizar el riesgo
if st.button('Analizar Riesgo'):
    # Crear un DataFrame con los datos de entrada en el formato original del entrenamiento
    # Es crucial que los nombres y el orden de las columnas coincidan con el X_train original
    input_data = pd.DataFrame([{
        'Edad': edad,
        'Plan': plan.lower(), # Asegurarse de que el plan esté en minúsculas como en el entrenamiento
        'Uso_Mensual_Km': uso_mensual_km,
        'Soporte_Tickets': soporte_tickets,
        'Gasto_Promedio': gasto_promedio,
        'Region': region,
        'Dias_Antiguedad': 365 # Valor por defecto, ya que no se ingresa en la UI. Ajusta si necesitas mayor precisión.
    }])

    # Reordenar las columnas para que coincidan con el orden de `X_train` antes del preprocesamiento
    # Esto es crucial para que el preprocesador funcione correctamente.
    original_columns_order = ['Edad', 'Plan', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Region', 'Dias_Antiguedad']
    input_data = input_data[original_columns_order]

    # Aplicar la transformación usando el preprocesador cargado
    input_data_prepared = preprocessor.transform(input_data)

    # Realizar la inferencia
    prediction = model.predict(input_data_prepared)[0]
    prediction_proba = model.predict_proba(input_data_prepared)[0]

    st.markdown("--- ")
    st.subheader('Resultado del Análisis:')

    # Mostrar el mensaje de alerta y la probabilidad
    if prediction == 1:
        st.error(f'**¡ALERTA! Alto Riesgo de Cancelación.**')
        st.write(f'Probabilidad de Churn: **{prediction_proba[1]*100:.2f}%**')
    else:
        st.success(f'**Cliente Estable.**')
        st.write(f'Probabilidad de No Churn: **{prediction_proba[0]*100:.2f}%**')
