import streamlit as st
import pandas as pd
import joblib

def normalisasi(x):
    data_test = pd.read_csv('AusAntidiabeticDrug.csv')
    cols = ['y']
    # date = data_test['noted_date'].mean()
    date = data_test['y'].mean()
    x.insert(1,date)
    # x.insert(1,temp)
    df = pd.DataFrame(x,columns=cols)
    print(df)
    # return df
    data_test = data_test.append(other = df,ignore_index = True)
    # data_test = data_test.astype(float)
    return joblib.load('Mm.sav').fit_transform(data_test)
def model(x):
    # print()
    return joblib.load('gpr.pkl').predict(x)
