# import libary 
import streamlit as st
import dataset
import time
import datetime
import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import streamlit.components.v1 as components
# pige title
st.set_page_config(
    page_title="",
    page_icon="https://cdn-icons-png.flaticon.com/128/254/254207.png",
)

    # 0 = tidak ada penyakit hepa
    # 1 = ada penyakit hepa

# hide menu
hide_streamlit_style = """



<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

"""


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">', unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown(' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="https://github.com/maulanamaib/streamlit_hepa.git" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> GitHub</button></a><a href="https://maulanamaib.github.io/datamining/intro.html" target="_blank"><button  style="border-radius: 12px;position: relative; top:50%;"><i style="color: orange" class="fa fa-book"></i> Jupyter</button></a></div>', unsafe_allow_html=True)
 

# colum = st.columns((0.1,10,1.5))
# url = 'https://github.com/maulanamaib/streamlit_hepa.git'

# if colum[1].button('GitHub'):
#     webbrowser.open_new_tab(url)

# link = 'https://maulanamaib.github.io/datamining/intro.html'

# if colum[2].button('Jupyter'):
    # webbrowser.open_new_tab(link)
# colum = st.columns((0.1,10,1.5))
# github= colum[1].button("check out this [link]()")
# jupyter = colum[2].button("")






# insialisasi web
# tab1, tab2, tab3, tab4 = st.tabs(["Form", "Normalisasi", "Model", "deskripsi"])
tab1, tab2= st.tabs(["Input","temp"])

with tab1:
    kolom = st.columns((1 , 1.5))   
    
#     url = 'https://github.com/maulanamaib/streamlit_hepa.git'

#     if kolom[1].button('GitHub'):
#         webbrowser.open_new_tab(url)

   
  
    # home = kolom[1].button('Home')
#     about = kolom[2].button('About')

   
#     kolom[4].button('Click Me!', 'https://maulanamaib.github.io/datamining/intro.html')

#     link = 'https://maulanamaib.github.io/datamining/intro.html'

#     if kolom[4].button('Jupyter'):
#         webbrowser.open_new_tab(link)

    # home page
#     if home==False and about==False or home==True and about==False:
        
    st.write("")
    st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Prediksi Penyakit Hepa</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>", unsafe_allow_html=True)

    # col1 = st.columns(1,1)
    # with col1:
    date = st.date_input("Date",datetime.date(1991,7,1))
    # with col2:
    #     temp = st.number_input("INPUT",max_value=100)
    # with col3:    
    #     jk = st.selectbox("Jenis Kelamin",('Laki-laki','Perempuan'))

    # bp = st.selectbox("Golongan Darah",("A","B","AB","O"))
    # col4,col5 =st.columns(2)
    # with col4:
    #     alt = st.number_input("masukkan nilai ALT", min_value=0 ,max_value=1000000000000)
    # with col5:
    #     ast = st.number_input("masukkan nilai AST",min_value=0,max_value=10000000000000)
        # col5, col6, col7, col8 = st.columns(4)
        # with col5:
        #     prot = st.number_input("Masukkan nilai prot")
        # with col6:
        #     alb = st.number_input("MAsukkan nilai alb")
        # with col7:
        #     alp = st.number_input("Masukkan nilai alp")
        # with col8:
        #     bil = st.number_input("Masukkan nilai bil")
        # col9, col10, col11, col12 = st.columns(4)
        # with col9:
        #     che = st.number_input("Masukkan nilai che")
        # with col10:
        #     chol = st.number_input("Masukkan nilali chol")
        # with col11:
        #     crea = st.number_input("Masukkan nilai crea")
        # with col12:
        #     a = st.number_input("masukkan a")
        
        # b = st.number_input("masukkan b")
        #    Centering Butoon 
    columns = st.columns((2, 0.6, 2))
    submit = columns[1].button("Submit")
    # if sumbit and nama != '' and jk != '' and bp != 0 and umur != 0  and ast != 0 and alt != 0:
    if submit:
            # cek jenis kelamin
            #0 = laki-laki
            #1 = perempuan
        # if jk == 'Laki-laki':
        #     jk = 0
        # else:
        #     jk = 1
            # normalisasi data
        data = dataset.normalisasi([date])
            # data = dataset.normalisasi([10,21,1,3])
            # prediksi data
        prediksi = dataset.model(data)  
        # print(prediksi)  
        if prediksi==date:
            st.write('benar')
        else:
            st.write('salah')
            # st.success("Hasil Prediksi : "+nama+" dengan golongan darah "+bp+" sehat!!")
            # cek prediksi
        # with st.spinner("Tunggu Sebentar Masih Proses..."):
        #     if prediksi[-1]== 0:
        #             # time.sleep(1)
        #         st.success("Hasil Prediksi : "+nama+" dengan golongna darah  "+bp+"  sehat!!")
                    
        #     elif prediksi[-1]==1:
        #         st.warning("Hasil Prediksi: "+nama+" kurang sehat")
        #     elif prediksi[-1]==2:
        #         st.warning("Hasil Prediksi: "+nama+" terkena hepatitis")
        #     elif prediksi[-1]==3:
        #         st.warning("Hasil Prediksi: "+nama+"  tekena fibrosis")
        #     elif prediksi[-1]==4:
        #         st.warning("Hasil Prediksi: "+nama+" terkena cirrhosis")          
#                 else :  
#                     time.sleep(1)
#                     st.warning("Hasil Prediksi : "+nama+"  dengan golongan darah "+bp+" Kemungkinan terkena penyakit hepa")

