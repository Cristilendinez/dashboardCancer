import psycopg2
from CRUD_bbdd import crud
# Components Pkgs
import streamlit as st

import pandas as pd
import numpy as np

# BBDD

st.title("Manejo de base de datos - BBDD 	🔍")

st.caption("Desarrollado por Cristina Lendinez y Maria Fernanda Mendoza")

# ==================================================================
df = pd.read_csv("./data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1) # Eliminar la columna 'id', 'Unnamed: 32'
columns = df.columns.values.tolist()

# ==================================================================

nameBBDD = st.text_input('Indique el nombre deseado para la base de datos:', '')

if st.button('CREAR BBDD', key = 'CREAR_BBDD'):
    # crud.createDatabase(nameBBDD)
    st.info("Proxima version")

# ==================================================================
nameTabla = st.text_input('Indique el nombre deseado para la tabla de datos:', '')
 
if st.button('CREAR TABLA', key = 'CREAR_TABLA'):
    # crud.createTabla(nameBBDD, nameTabla)
    st.info("Proxima version")

# ==================================================================
lista = st.text_input('Inserte los siguientes datos:', '') 

lista = [('M',17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189),('M',20.57,17.77,132.9,1326.0,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956.0,0.1238,0.1866,0.2416,0.186,0.275,0.08902),('M',20.29,14.34,135.1,1297.0,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575.0,0.1374,0.205,0.4,0.1625, 0.2364,0.07678),('M',15.85,23.95,103.7,782.7,0.08401,0.1002,0.09938,0.05364,0.1847,0.05338,0.4033,1.078,2.903,36.58,0.009769,0.03126,0.05051,0.01992,0.02981,0.003002,16.84,27.66,112.0,876.5,0.1131,0.1924,0.2322,0.1119,0.2809,0.06287)]

if st.button('INSERTAR DATOS', key = 'INSERTAR DATOS'):
    # for i in lista:
        # crud.insertarDatos(nameTabla, i)
    st.info("Proxima version")

# ==================================================================
opcions = ['-'] + columns
column1 = st.selectbox('Indique la columna de la que desee extraer datos:', opcions, key = 'MOSTRAR')

if column1 in columns:
    low = df[column1].min().tolist()
    high = df[column1].max().tolist()
    low_1 = low + 1
    high_1 = high -1

    val1, val2 =  st.slider('Seleccione rango de valores', low, high, (low_1, high_1), key = 'Rango_mostrar')           

if st.button('MOSTRAR DATOS', key = 'MOSTRAR DATOS'):
    
    # st.write(crud.mostrar(nameTabla, column1, val1, val2))
    st.info("Proxima version")

# ==================================================================

if st.button('ELIMINAR DATOS', key = 'ELIMINAR DATOS'):
    # crud.eliminar(nameTabla)
     st.info("Proxima version")

