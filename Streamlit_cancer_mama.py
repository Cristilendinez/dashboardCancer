import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

import io


# Pre-Modeling Tasks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


# Evaluation and comparision of all the models
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,auc,f1_score
from sklearn.metrics import precision_recall_curve,roc_curve

st.title('ESTUDIO SOBRE EL CANCER EN MAMA')

st.markdown('Una vez definida nuestra estrategia de trabajo, comenzaremos con los datos y aplicaremos  los modelos considerados.')

df = pd.read_csv("./data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1) # Eliminar la columna 'id', 'Unnamed: 32'
columns = df.columns.values.tolist()

st.header('Exploratory Data Analisys - EDA', anchor=None)

info = st.checkbox('Información complementaria del dataset')
if info:
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write('Se tiene un total de ', df.shape[0] , 'filas y de ', df.shape[1], ' columnas. Sin valores nulos')

# Cabeza/Cola/Muestra

df_vis = ('Inicio','Final', 'Muestra')
radio_0 = st.radio('Visulizar dataframe:', df_vis)
n_filas = st.slider('Cuantas filas deseas visualizar?', 0, len(df), 5)

if radio_0 == df_vis[0]:
    st.dataframe(df.head(n_filas))
elif radio_0 == df_vis[1]:
    st.dataframe(df.tail(n_filas))
else:
    st.dataframe(df.sample(n_filas))

cor_matrix = df.corr().abs()

info0 = st.checkbox('Tabla de correlación')
if info0:
    rango = st.slider('Seleccione el rango de correlación deseado', value = [0.5,1.0])
    
    print(rango[0],type(rango[0]))
    st.write(cor_matrix.style.highlight_between(left=rango[0], right=rango[1],inclusive = 'left'))    

# Mapa calor

info1 = st.checkbox('Mapa de calor')
if info1:
    fig, ax = plt.subplots(figsize=(20,10))    
    sns.heatmap(cor_matrix, annot=True)
    st.pyplot(fig)


# Correlación por columnas
info2 = st.checkbox('Gráfico de correlación de entre una selección de columnas')
if info2:
    opcion1 = st.multiselect('Escoja las columnas que desea correlacionar son: ', columns)
    st.write('Columnas seleccionadas: ', opcion1)

    radio_1 = st.radio('Tener en cuenta los diagnosticos:', ('Malos','Buenos', 'Todos'))

    if radio_1 == 'Malos':
        fig = px.scatter_matrix(df.loc[df['diagnosis']=='M'], dimensions = opcion1, color = 'diagnosis', symbol = 'diagnosis', title = 'Scatter matrix of cancer mama', labels={col:col.replace('_', ' ') for col in opcion1})    
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)
    elif radio_1 == 'Buenos':
        fig = px.scatter_matrix(df.loc[df['diagnosis']=='B'], dimensions = opcion1, color = 'diagnosis', symbol = 'diagnosis', title = 'Scatter matrix of cancer mama', labels={col:col.replace('_', ' ') for col in opcion1})    
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)
    else:    
        fig = px.scatter_matrix(df, dimensions = opcion1, color = 'diagnosis', symbol = 'diagnosis', title = 'Scatter matrix of cancer mama', labels={col:col.replace('_', ' ') for col in opcion1})    
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)


st.header('Estadística descriptiva', anchor=None)

# Descripción por columnas
info3 = st.checkbox('Descripción y gráficos por columnas')

if info3:
    opcion2 = st.multiselect('Escoja las columnas que desea evaluar: ', columns)

    if len(opcion2) == 0:
        st.write('Seleccione como mínimo una variable') 
    
    elif len(opcion2) == 1:
        st.write('Columnas seleccionadas: ', opcion2)
        df_op2 = df[opcion2]
        st.write(df[opcion2].describe().T)

        opcion3 = st.selectbox('Gráfico:', '-', 'Histogram')

        radio1 = st.radio('Tener en cuenta todos tipos de diagnostico:', ('Si','No'))

        if radio_1 == 'No':         
            fig = px.histogram(df, x = opcion2)
            st.plotly_chart(fig, theme=None, use_container_width=True)
        else:    
            fig = px.histogram(df, x = opcion2, color = 'diagnosis')
            st.plotly_chart(fig, theme=None, use_container_width=True)

    else: 
        st.write('Columnas seleccionadas: ', opcion2)
        df_op2 = df[opcion2]
        st.write(df[opcion2].describe().T)
        
        opcion3 = st.selectbox('Gráfico:', ('Box Plots','Histogram','Distplots'))

        if opcion3 == 'Box Plots':
            opcion4 = st.selectbox('Escoja la x: ', opcion2)

            fig = px.box(df, x = 'diagnosis', y = opcion4, points="all")
            st.plotly_chart(fig, theme=None, use_container_width=True)

        elif opcion3 == 'Histogram':
            opcion4 = st.selectbox('Escoja una variable: ', opcion2)

            fig = px.histogram(df, x = opcion4, color = 'diagnosis')
            st.plotly_chart(fig, theme=None, use_container_width=True)               
        
        elif opcion3 == 'Distplots':
            opcion4 = st.selectbox('Escoja una variable para el eje de las abscisas: ', opcion2)
            opcion5 = opcion2
            print(opcion4, type(opcion4))
            print(opcion5, type(opcion5))
            opcion6 = st.selectbox('Escoja una variable para el eje de las ordenadas: ', opcion5)
            
            fig = px.histogram(df, x = opcion4, y = opcion6, color="diagnosis", marginal="box")
            st.plotly_chart(fig, theme=None, use_container_width=True)


st.header('Algoritmos de clasificación', anchor=None)

radio = st.radio('Escoje el tipo de algoritmo que deseas implementar:', ('Regresión','Clasificación', 'Clustering', 'Association'))

# X = 
# y = 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if radio == 'Regresión':
    opciones = st.multiselect('Tipos: ', ['Lineal simple', 'Lineal multiple','Lineal polinomial', 'Supot Vector Regression - SVR', 'Decission Tree Regression', 'Ramdom Forest Regression'])

    st.write('Has seleccionado la Regresión ', opciones)
    if opciones == 'Lineal simple':
        pass

        # lr = linear_model.LinearRegression()
        # lr.fit(X_train, y_train)
        # y_pred = lr.predict(X_test)
        # st.write(f"Precisión del modelo: {lr.score(X_train, y_train)}")

        # plt.scatter(X_test, y_test)
        # plt.plot(X_test, y_pred, color='red', linewidth=3)
        # plt.title("Regresión Lineal Simple")
        # plt.xlabel("Número de habitantes")
        # plt.ylabel("Valor medio")
        # plt.show()

    elif opciones == 'Lineal multiple':

        pass

        # lr = linear_model.LinearRegression()
        # lr.fit(X_train, y_train)
        # y_pred = lr.predict(X_test)
        # st.write(f"Precisión del modelo: {lr.score(X_train, y_train)}")

        # plt.scatter(X_test, y_test)
        # plt.plot(X_test, y_pred, color='red', linewidth=3)
        # plt.title("Regresión Lineal Simple")
        # plt.xlabel("Número de habitantes")
        # plt.ylabel("Valor medio")
        # plt.show()

    elif opciones == 'Lineal polinomial':
        pass

    elif opciones == 'Supot Vector Regression - SVR':

        pass

    elif opciones == 'Decission Tree Regression':

        pass

    else:

        pass

elif radio == 'Clasificación':
    opciones = st.multiselect('Tipos: ', ['Logistic regrassion', 'K-Nearest Neighbors', 'Supot Vector Machine - SVM', 'Naives Bayes', 'Decission Tree Classification', 'Random Forest Classification'])

    st.write('Has seleccionado el algoritmo de Clustering ', opciones)

    if opciones == 'Logistic regrassion':
        pass

    elif opciones == 'K-Nearest Neighbors':

        pass

    elif opciones == 'Supot Vector Machine - SVM':
        pass

    elif opciones == 'Naives Bayes':

        pass

    elif opciones == 'Decission Tree Regression':

        pass
    elif opciones == 'Decission Tree Classification':
        pass

    else:

        pass

elif radio == 'Clustering':
    opciones = st.multiselect('Tipos: ', ['K-Means Clustering', 'Hierarchical Clustering'])

    st.write('Has seleccionado el algoritmo de clustering ', opciones)

    if opciones == 'K-Means Clustering':
        pass

    else:

        pass
else:
    opciones = st.multiselect('Tipos: ', ['Association Rule Learning', 'Apriori Algoritm', 'Market Basket Analysis'])

    st.write('Has seleccionado el algoritmo de asociación ', opciones)

    if opciones == 'Association Rule Learning':
        pass
    
    elif opciones == 'Apriori Algoritm':
        pass

    else:

        pass



st.header('Validación', anchor=None)

