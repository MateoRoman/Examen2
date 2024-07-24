import numpy as np
import pickle
import streamlit as st

# Cargar el modelo y el scaler
MODEL_PATH = 'svm_rbf_model.pkl'
SCALER_PATH = 'scaler.pkl'

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

# Función para hacer predicciones
def model_prediction(x_in, model, scaler):
    print("Datos de entrada:", x_in)  # Depuración
    x_scaled = scaler.transform(np.asarray(x_in).reshape(1, -1))
    print("Datos escalados:", x_scaled)  # Depuración
    preds = model.predict(x_scaled)
    print("Predicción:", preds)  # Depuración
    return preds

def main():
    # Título
    st.title("SISTEMA DE RECOMENDACIÓN PARA SALARIO")

    # Entrada de datos
    st.markdown("Ingrese los valores de las características:")
    age = st.text_input("Edad:")
    education_num = st.text_input("Número de años de educación:")
    capital_gain = st.text_input("Ganancia de capital:")
    capital_loss = st.text_input("Pérdida de capital:")
    hours_per_week = st.text_input("Horas por semana:")
    
    # Campos adicionales
    workclass = st.text_input("Clase de trabajo:")
    education = st.text_input("Educación:")
    marital_status = st.text_input("Estado civil:")
    occupation = st.text_input("Ocupación:")
    relationship = st.text_input("Relación:")
    race = st.text_input("Raza:")
    sex = st.text_input("Sexo:")
    native_country = st.text_input("País de origen:")
    fnlwgt = st.text_input("Peso final:")
    
    # Botón para hacer la predicción
    if st.button("Predicción"):
        try:
            # Asegúrate de que el orden de las características sea el mismo que usaste para entrenar el modelo
            x_in = [
                age, education_num, capital_gain, capital_loss, hours_per_week,
                workclass, education, marital_status, occupation, relationship,
                race, sex, native_country, fnlwgt
            ]
            
            # Convertir todos los valores a float
            x_in = [float(val) for val in x_in]
            prediction = model_prediction(x_in, model, scaler)
            st.success(f'El salario predicho es: {prediction[0]}')
        except ValueError as e:
            st.error(f"Error en la predicción: {e}")
        except Exception as e:
            st.error(f"Error inesperado: {e}")

if __name__ == '__main__':
    main()
