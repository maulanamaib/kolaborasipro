import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import model2


# Mengimpor model dan text dari file yang telah diekspor sebelumnya
svr = joblib.load('modelSVM.pkl')
df = pd.read_csv('AusAntidiabeticDrug.csv')
scaler = joblib.load('scaler.pkl')
dataset_test = pd.read_csv('data-test.csv')
def ramal(n_pred, dataset_test, tahun):
    tahun = tahun[0]
    last = dataset_test.tail(1)
    fitur = last.values
    n_fit = len(fitur[0])
    fiturs = np.zeros((n_pred, n_fit))
    tahuns = np.zeros(n_pred)
    preds = np.zeros(n_pred)
    for i in range(n_pred):
        if i == 0:
            fitur = fitur[:, 1:n_fit]
            y_pred=svr.predict(fitur)
            new_fit = np.array(fitur[0])
            new_fit = np.append(new_fit,y_pred)
        else:
            fitur = fiturs[i-1][1:n_fit]
            y_pred=svr.predict([fitur])
            new_fit = np.array(fitur)
            new_fit = np.append(new_fit,y_pred)
        preds[i] = y_pred
        fiturs[i,:] = new_fit 
        tahun += 1
        tahuns[i] = tahun
#     print(preds)
#     print(fiturs)
    return preds, tahuns.astype(int)






# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>Forecasting Temperature Anomalies Data Time Series</h1>", unsafe_allow_html=True)
# st.title('Forecasting Temperature Anomalies Data Time Series')


# Define a range for the slider
min_value = 0
max_value = 60

# Create a slider widget
selected_value = st.slider('Select a value', min_value, max_value)

# Display the selected value
st.write('You selected:', selected_value)

# Tombol untuk memprediksi
if st.button('Run'):
    if selected_value:
        col1, col2 = st.columns(2)
        tahun_terakhir = df['y'].tail(1).values
        pred, tahun = ramal(selected_value,dataset_test, tahun_terakhir)
        # print(pred)
        # print(tahun)
        reshaped_data = pred.reshape(-1, 1)
        original_data = scaler.inverse_transform(reshaped_data)
        # print(original_data)
        df_pred = pd.DataFrame({'ds': tahun, 'y': pred})
        # st.dataframe(df_pred)
        # st.write('Hasil Prediksi:', prediction)

        # Plot data df
        fig, ax = plt.subplots()
        ax.plot(df['ds'], df['y'], label='Data Awal')

        # Plot data df_pred
        ax.plot(df_pred['ds'], df_pred['y'], label='Prediksi')

        # Menghubungkan plot terakhir data awal dengan plot awal data prediksi
        last_ds = df['ds'].iloc[-1]
        ax.plot([last_ds, df_pred['ds'].iloc[0]], [df['y'].iloc[-1], df_pred['y'].iloc[0]], 'k--')

        # Konfigurasi plot
        ax.set_xlabel('Time')
        ax.set_ylabel('Drug')
        ax.set_title('Perbandingan Data Awal dan Prediksi')
        ax.legend()

        # Tampilkan plot
        # st.pyplot(fig)
        with col1:
            st.dataframe(df_pred)
        with col2:
            st.pyplot(fig)
    else:
        st.write('Masukkan teks terlebih dahulu')
# print(tahun_terakhir)