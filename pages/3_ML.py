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

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC


# Evaluation and comparision of all the models
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,auc,f1_score
from sklearn.metrics import precision_recall_curve,roc_curve


import warnings
warnings.filterwarnings("ignore")

# ML

st.header('Aprendizaje automático -  ML', anchor=None)

# DATOS 

df = pd.read_csv("./data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1) # Eliminar la columna 'id', 'Unnamed: 32'

df_x = df.drop(['diagnosis'], axis = 1)
keys_x = df_x.keys().to_list()

X_lis = st.sidebar.multiselect('Seleccione las variable para entrenar el modelo de predicción: ', options = keys_x, default = keys_x )

df_y = df['diagnosis']

st.markdown('Variables selecionadas:')
st.dataframe(df_x[X_lis])

data_test = st.slider('Indique el porcentaje de datos para la validación:', min_value=60, max_value=100, step=10, format=None, key=None, disabled=False, label_visibility="visible")
data_test = round(1-(data_test/100),2)

# Graficos Funciones
def grafico_Scatter(X_train, X_test, y_train, y_test, Xc, Yc):

    df_training = pd.DataFrame(X_train, columns = [Xc])
    df_training[Yc] = y_train
    df_training['Tipo'] = 'Training'
    df_test = pd.DataFrame(X_test, columns = [Xc])
    df_test[Yc] = y_test
    df_test['Tipo'] = 'Test'       

    df1 = pd.concat([df_training, df_test])  

    fig = px.scatter(df1, x= Xc, y= Yc, color="Tipo")
    st.plotly_chart(fig, use_container_width=True)
def Plot_3D(X, X_test, y_test, clf):
            
    # Specify a size of the mesh to be used
    mesh_size = 5
    margin = 1

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
            
    # Calculate predictions on grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot with predictions: 'principal component 1', 'principal component 2', 'principal component 3'
    fig = px.scatter_3d(x=X_test['principal component 1'], y=X_test['principal component 2'], z=y_test, 
                     opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(#title_text="Scatter 3D Plot with SVM Prediction Surface",
                      paper_bgcolor = 'white',
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0', 
                                              )))
    # Update marker size
    fig.update_traces(marker=dict(size=1))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='SVM Prediction',
                              colorscale='RdBu', showscale=False, 
                              contours = {"z": {"show": True, "start": 0.2, "end": 0.8, "size": 0.05}}))
    fig.show()



tab1, tab2 = st.tabs(['Regresión','Clasificación'])
with tab1:
    opciones = st.selectbox('Tipos: ', ['Lineal simple', 'Lineal multiple','Lineal polinomial'])

    st.write('Has seleccionado la Regresión ', opciones)

    if opciones == 'Lineal simple':
        from sklearn import datasets, linear_model
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox('Selecione variable 1:', options = X_lis, key='R1')
        with col2:
            var2 = st.selectbox('Selecione variable 2:', options = X_lis, key='R2')

        X = np.array(df_x.loc[:,var1]).reshape(-1, 1)
        y = np.array(df_x.loc[:,var2])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = data_test)

        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        st.write('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Valor de la pendiente o coeficiente "a":')
            st.write(lr.coef_)
        with col2:
            st.write('Valor de la intersección o coeficiente "b":')
            st.write(lr.intercept_)
            
        st.write(f'La ecuación del modelo es igual a: \n y = {lr.coef_}x + {lr.intercept_}')

        grafico_Scatter(X_train, X_test, y_train, y_test, var1, var2)

        st.metric(label='Presición del modelo', value = lr.score(X_train, y_train))

    elif opciones == 'Lineal multiple':

        from sklearn.model_selection import train_test_split
        from sklearn import linear_model

        targ = st.selectbox('Seleccione una variable como target:', X_lis)

        resto = []
        for i in X_lis:
            if i != targ:
                resto.append(i)
        
        X = np.array(df_x.loc[:,resto])
        y = np.array(df_x.loc[:,targ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = data_test)    

        lr_multiple = linear_model.LinearRegression()   
        lr_multiple.fit(X_train, y_train) 
        y_pred = lr_multiple.predict(X_test)

        st.write('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Valores de la pendiente o coeficientes "a":')
            st.write(lr_multiple.coef_)
        with col2:
            st.write('Valor de la intersección o coeficiente "b":')
            st.write(lr_multiple.intercept_)

        st.metric(label='Presición del modelo', value = lr_multiple.score(X_train, y_train))

    elif opciones == 'Lineal polinomial':
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn import linear_model
        
        col1, col2 = st.columns(2)
        with col1:
            var3 = st.selectbox('Selecione variable 1:', options = X_lis, key='R3')
        with col2:
            var4 = st.selectbox('Selecione variable 2:', options = X_lis, key='R4')

        X = np.array(df_x.loc[:,var3]).reshape(-1, 1)
        y = np.array(df_x.loc[:,var4])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = data_test)

        poli_reg = PolynomialFeatures(degree= 2)
        pr = linear_model.LinearRegression()
        pr.fit(X_train, y_train)   
        y_pred = pr.predict(X_test)


        st.write('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('Valor del coeficiente "a":')
            st.write(pr.coef_[:])

        grafico_Scatter(X_train, X_test, y_train, y_test, var3, var4)

        st.metric(label='Presición del modelo', value = pr.score(X_train, y_train))

    else:

        pass

with tab2:  #  == 'Clasificación': 

    choice = st.checkbox('Preprocesado de los datos:')
    if choice:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import StandardScaler

        radio = st.radio('Escoge uno:', ['MinMaxScaler', 'StandardScaler'], key='prepro')
        if radio == 'MinMaxScaler':
            minmax = MinMaxScaler()
            df_x[X_lis] = minmax.fit_transform(df_x[X_lis])
        else:
            ss = StandardScaler()
            df_x[X_lis]= ss.fit_transform(df_x[X_lis])

        df_y = OneHotEncoder().fit_transform(df[['diagnosis']]).toarray()[:, 1]

    st.dataframe(df_x)
    
    X = np.array(df_x.loc[:, X_lis])
    y = np.array(df_y)    

    var3 = X_lis
    var4 = 'diagnosis'

    choice_PCA = st.checkbox('Implementar PCA:')

    
    if choice_PCA:

        tab_a, tab_b = st.tabs(['2-Components', '3-Components'])

        with tab_a:
            from sklearn.decomposition import PCA   
            algorithm_pca = PCA(n_components=2)
            principalComponents_breast_2= algorithm_pca.fit_transform(X)   
            principal_breast_Df_2 = pd.DataFrame(data = principalComponents_breast_2, columns = ['principal component 1', 'principal component 2'])

            principal_breast_Df_2['Tipo'] = y

            st.dataframe(principal_breast_Df_2)
            st.write('Explained variation per principal component: {}'.format(algorithm_pca.explained_variance_ratio_))    
            X_train, X_test, y_train, y_test = train_test_split(principalComponents_breast_2, y, test_size = data_test)

            fig = px.scatter(principal_breast_Df_2, x= 'principal component 1', y= 'principal component 2', color="Tipo")
            st.plotly_chart(fig, use_container_width=True)

        with tab_b:

                from sklearn.decomposition import PCA   
                algorithm_pca = PCA(n_components=3)
                principalComponents_breast_3 = algorithm_pca.fit_transform(X)   
                principal_breast_Df_3 = pd.DataFrame(data = principalComponents_breast_3, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

                principal_breast_Df_3['Tipo'] = y

                st.dataframe(principal_breast_Df_3)
                st.write('Explained variation per principal component: {}'.format(algorithm_pca.explained_variance_ratio_))    
                X_train, X_test, y_train, y_test = train_test_split(principalComponents_breast_3, y, test_size = data_test)

                fig = px.scatter_3d(principal_breast_Df_3, x= 'principal component 1', y= 'principal component 2', z = 'principal component 3', color="Tipo")
                st.plotly_chart(fig, use_container_width=True)


    else:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = data_test)


    opciones = st.selectbox('Tipos: ', ['-','Logistic regression', 'K-Nearest Neighbors', 'Supot Vector Machine - SVM', 'Naives Bayes', 'Decission Tree Classification', 'Random Forest Classification'])
    st.write('Has seleccionado el algoritmo de clasificación: ', opciones)


    if opciones == 'Logistic regression':
        from sklearn.linear_model import LogisticRegression

        algoritmo = LogisticRegression()
        algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        col_a, col_b = st.columns(2)
        with col_a:
            df_comp = pd.DataFrame(list(zip(y_test,y_pred)), columns= ['y_test','y_pred'])
            st.dataframe(df_comp)

        with col_b:
            st.write('Recuento de valores de predicción:')
            st.write('0: Benigno     /    1: Maligno')
            st.dataframe(pd.concat([df_comp.y_pred.value_counts(),df_comp.y_test.value_counts()],axis = 1))
            df_comp.y_test.value_counts()
            # st.write(f'La presición del modelo es {round((accuracy_score(y_test,y_pred))*100,1)}%')
            st.metric(label='Presición del modelo', value = round((accuracy_score(y_test,y_pred))*100,1))

    elif opciones == 'K-Nearest Neighbors':
        from sklearn.neighbors import KNeighborsClassifier

        algoritmo = KNeighborsClassifier()
        algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        col_a, col_b = st.columns(2)
        with col_a:
            df_comp = pd.DataFrame(list(zip(y_test,y_pred)), columns= ['y_test','y_pred'])
            st.dataframe(df_comp)

        with col_b:
            st.write('Recuento de valores de predicción:')
            st.write('0: Benigno     /    1: Maligno')
            st.dataframe(pd.concat([df_comp.y_pred.value_counts(),df_comp.y_test.value_counts()],axis = 1))
            df_comp.y_test.value_counts()
            st.metric(label='Presición del modelo', value = round((accuracy_score(y_test,y_pred))*100,1))

    elif opciones == 'Supot Vector Machine - SVM':
        from sklearn import svm

        tipoK= st.radio('Tipo de kernel: ', ('Lineal', 'Polinomial', 'rbf'), key='tipok')

        if tipoK == 'Lineal':
            algoritmo = svm.SVC(kernel='linear', probability=True)
        elif tipoK == 'Polinomial':
            algoritmo = svm.SVC(kernel='poly', probability=True)
        else:
            algoritmo = svm.SVC(kernel='rbf', probability=True)
        
        clf = algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        col_a, col_b = st.columns(2)

        with col_a:
            df_comp = pd.DataFrame(list(zip(y_test,y_pred)), columns= ['y_test','y_pred'])
            st.dataframe(df_comp)

        with col_b:
            st.write('Recuento de valores de predicción:')
            st.write('0: Benigno     /    1: Maligno')
            st.dataframe(pd.concat([df_comp.y_pred.value_counts(),df_comp.y_test.value_counts()],axis = 1))
            df_comp.y_test.value_counts()
            st.metric(label='Presición del modelo', value = round((accuracy_score(y_test,y_pred))*100,1))

        # if choice_PCA:

        #     with tab_a:
        #         principal_breast_Df_2 
        #         X_2=principal_breast_Df_2[['principal component 1', 'principal component 2']]
        #         y=principal_breast_Df_2['Tipo'].values
        #         fig = Plot_3D(X_2, X_test, y_test, clf)
        #         st.plotly_chart(fig, theme="streamlit", use_conatiner_width=True)
            
        #     with tab_b:
        #         principal_breast_Df_3
        #         X_3=principal_breast_Df_3[['principal component 1', 'principal component 2', 'principal component 3']]
        #         y=principal_breast_Df_3['Tipo'].values
        #         fig = Plot_3D(X_3, X_test, y_test, clf)
        #         st.plotly_chart(fig, theme="streamlit", use_conatiner_width=True)

    elif opciones == 'Naives Bayes': # GaussianNB - Naives Bayes
        from sklearn.naive_bayes import GaussianNB

        algoritmo = GaussianNB()
        algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        col_a, col_b = st.columns(2)
        with col_a:
            df_comp = pd.DataFrame(list(zip(y_test,y_pred)), columns= ['y_test','y_pred'])
            st.dataframe(df_comp)

        with col_b:
            st.write('Recuento de valores de predicción:')
            st.write('0: Benigno     /    1: Maligno')
            st.dataframe(pd.concat([df_comp.y_pred.value_counts(),df_comp.y_test.value_counts()],axis = 1))
            df_comp.y_test.value_counts()
            # st.write(f'La presición del modelo es {round((accuracy_score(y_test,y_pred))*100,1)}%')
            st.metric(label='Presición del modelo', value = round((accuracy_score(y_test,y_pred))*100,1))

    elif opciones == 'Decission Tree Regression':

        pass
    elif opciones == 'Decission Tree Classification':
        from sklearn.tree import DecisionTreeClassifier

        algoritmo = DecisionTreeClassifier()
        algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        col_a, col_b = st.columns(2)
        with col_a:
            df_comp = pd.DataFrame(list(zip(y_test,y_pred)), columns= ['y_test','y_pred'])
            st.dataframe(df_comp)

        with col_b:
            st.write('Recuento de valores de predicción:')
            st.write('0: Benigno     /    1: Maligno')
            st.dataframe(pd.concat([df_comp.y_pred.value_counts(),df_comp.y_test.value_counts()],axis = 1))
            df_comp.y_test.value_counts()
            # st.write(f'La presición del modelo es {round((accuracy_score(y_test,y_pred))*100,1)}%')
            st.metric(label='Presición del modelo', value = round((accuracy_score(y_test,y_pred))*100,1))

    elif opciones == 'Random Forest Classification':

        from sklearn.ensemble import RandomForestClassifier

        algoritmo = RandomForestClassifier(n_estimators=100)
        algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        col_a, col_b = st.columns(2)
        with col_a:
            df_comp = pd.DataFrame(list(zip(y_test,y_pred)), columns= ['y_test','y_pred'])
            st.dataframe(df_comp)

        with col_b:
            st.write('Recuento de valores de predicción:')
            st.write('0: Benigno     /    1: Maligno')
            st.dataframe(pd.concat([df_comp.y_pred.value_counts(),df_comp.y_test.value_counts()],axis = 1))
            df_comp.y_test.value_counts()
            # st.write(f'La presición del modelo es {round((accuracy_score(y_test,y_pred))*100,1)}%')
            st.metric(label='Presición del modelo', value = round((accuracy_score(y_test,y_pred))*100,1))
    else:
        pass    


# st.header('Validación', anchor=None)