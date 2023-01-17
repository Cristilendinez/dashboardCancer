import streamlit as st

# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report


# EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import plotly.figure_factory as ff
import io

import codecs
from pandas_profiling import ProfileReport
import sweetviz as sv

# EDA


st.title("Analísis Exploratorio de Datos - EDA 	🔍")

st.caption("Desarrollado por Cristina Lendinez y Maria Fernanda Mendoza")

df = pd.read_csv("./data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1) # Eliminar la columna 'id', 'Unnamed: 32'
columns = df.columns.values.tolist()

st.write(f'En este apartado se pueden evaluar las características dimensionales detectadas en diferentes diagnosticos hechos del cancer. Este dataset consta de {df.shape[0]} diagnósticos y de {df.shape[1]} características. La finalidad será detectar los insigths que ayuden a predecir el diagnostico, si el cancer es maligno (M) o Benigno (B)')

menu_eda = [' - ','Analísis Exploratorio por Variable', 'Información complementaria del dataset', 'Correlación', 'Gráficos']

choice= st.sidebar.selectbox("Escoge el analísis a realizar:", menu_eda)


if choice == 'Analísis Exploratorio por Variable':

    st.header('Descripción estadística por variables')

    def st_display_sweetviz(report_html, width=1000, height= 500):
        report_file = codecs.open(report_html, 'r')
        page= report_file.read()
        components.html(page,width=width, height=height, scrolling=True)

    footer_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h5 class="white-text">About Streamlit EDA App</h5>
	          <p class="grey-text text-lighten-4">Using Streamlit,Pandas,Pandas Profile and SweetViz.</p>
	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Connect With Me</h5>
	          <ul>
	            <a href="https://facebook.com/jcharistech" target="_blank" class="white-text">
	            <i class="fab fa-facebook fa-4x"></i>
	          </a>
	          <a href="https://gh.linkedin.com/in/jesiel-emmanuel-agbemabiase-6935b690" target="_blank" class="white-text">
	            <i class="fab fa-linkedin fa-4x"></i>
	          </a>
	          <a href="https://www.youtube.com/channel/UC2wMHF4HBkTMGLsvZAIWzRg" target="_blank" class="white-text">
	            <i class="fab fa-youtube-square fa-4x"></i>
	          </a>
	           <a href="https://github.com/Jcharis/" target="_blank" class="white-text">
	            <i class="fab fa-github-square fa-4x"></i>
	          </a>
	          </ul>
	        </div>
	      </div>
	    </div>
	    <div class="footer-copyright">
	      <div class="container">
	      Made by <a class="white-text text-lighten-3" href="https://jcharistech.wordpress.com">Jesse E.Agbe & JCharisTech</a><br/>
	      <a class="white-text text-lighten-3" href="https://jcharistech.wordpress.com">Jesus Saves @JCharisTech</a>
	      </div>
	    </div>
	  </footer>
	"""
    if st.button("Generar Reporte"):
        st_display_sweetviz("SWEETVIZ_REPORT.html")

    report  = sv.analyze(df)
    report.show_html("SWEETVIZ_REPORT.html")

elif choice == 'Información complementaria del dataset': 

    st.header('Visualización del dataframe "Cancer de mama"')

    col1, col2 = st.columns(2)

    with col1:
    
        df_vis = ('Inicio','Final', 'Muestra')
        radio_0 = st.radio('Visulizar set:', df_vis)
        n_filas = st.number_input(label = 'Cuantas filas deseas visualizar?', min_value = 10, max_value= len(df), step=None, format=None, key=int, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    with col2:

        radio_1 = st.radio('Tener en cuenta los diagnosticos:', ('Malos','Buenos', 'Todos'))


    opcion1 = st.multiselect(label = 'Elimine las columnas que no desee evaluar en el dataframe: ', options = columns, default= columns, key='InfComp', help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility = "visible", max_selections=None)
        
       
    def B_M(df, radio_1):    
        if radio_1 == 'Malos':
            return df.loc[df['diagnosis']=='M']
        
        elif radio_1 == 'Buenos':
            return df.loc[df['diagnosis']=='B']

        return df

    df1 = B_M(df[opcion1], radio_1)

    def IniFinTo(df, radio_0):
        if radio_0 == df_vis[0]:
           return st.dataframe(df.head(n_filas))
        elif radio_0 == df_vis[1]:
            return st.dataframe(df.tail(n_filas))
        else:
            return st.dataframe(df.sample(n_filas))

    IniFinTo(df1, radio_0)

# Gráficos
elif choice == 'Gráficos': 

    st.header('Gráficos')

    opcion1 = columns

    def graf_histograma(df, opcion):
        fig = px.histogram(df, x = opcion, color = 'diagnosis')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    def graf_scatter(df, opcion):
        fig = px.scatter_matrix(df, dimensions = opcion, color = 'diagnosis', symbol = 'diagnosis', title = 'Scatter matrix of cancer mama', labels={col:col.replace('_', ' ') for col in opcion})    
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)

    def graf_linea(df, opcion):
        st.line_chart(df[opcion])

    def graf_PCA(df, opcion):
        fig = px.scatter_matrix(df, dimensions = opcion, color='diagnosis')
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    def graf_Theming(df, opcion):
        
        # print(opcion, type(opcion), 'diagnosis' in opcion)
        
        # if 'diagnosis' in opcion:
        #     op3 = opcion.remove('diagnosis')    
        #     print(op3, type(op3), 'diagnosis' in opcion)
        # else:
        #     op3 = opcion      
        op3 = opcion 

        col_a, col_b = st.columns(2)

        with col_a: 
            x1 = st.selectbox('Escoja x: ', op3, key = 'x1')
            print(type(x1), x1)
            # op4 = op3.remove(x1)
            x2 = st.selectbox('Escoja y: ', op3, key = 'x2')
            # op5 = op4.remove(x2)
            x3 = st.selectbox('Escoja z: ', op3, key = 'x3')
            # op6 = op5.remove(x3)
            x4 = st.selectbox('Escoja t: ', op3, key = 'x4')

        with col_b:

            low1 = df[x1].min()
            high1 = df[x1].max()

            low2 = df[x2].min()
            high2 = df[x2].max()

            # print(low1, type(low1[0]), high1, type(high1))

            # # low_1, high_1 =
            # st.slider('Seleccione rango de valores', low1[0], high1[0], (low1[0], high1[0]))
            # # low_2, high_2 = st.slider('Seleccione rango de valores', low2, high2, (low2, high2))
        

        # mask1 = (df[x1] > low_1) & (df[x1] < high_1)
        # mask2 = (df[x2] > low_2) & (df[x2] < high_2)

        # fig = px.scatter(df[mask1 and mask2], x = x1, y = x2, size = x3, color= "diagnosis", hover_name = x4, log_x=True, size_max=60)
        # st.plotly_chart(fig, theme=None, use_container_width=True)

    def graf_colorscale(df, x1, x2, x3):
        fig = px.scatter(df, x = x1, y = x2, color= x3, color_continuous_scale="reds")
        st.plotly_chart(fig, theme="streamlit", use_conatiner_width=True)

    def graf_barChar(df, opcion):
        st.bar_chart(df[opcion])

    def graf_Scatter3D(df, x1, y1, z1):
        fig = px.scatter_3d(df, x=x1, y=y1, z=z1,
              color='diagnosis')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8   = st.tabs(['Histograma','Scatter', 'Lineal', 'PCA', "Theming", "Colorscale", 'BarChar', '3D Scatter'])
    
    with tab1:
        x = st.selectbox('',opcion1)
        graf_histograma(df, x)
    with tab2:
        op = st.multiselect('Seleccione variables:', options = opcion1, default=None,  key="tab2", help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible", max_selections=None)
        if len(op) !=  0:
            graf_scatter(df, op)
    with tab3:
        op1 = st.multiselect('Seleccione variables', options = opcion1, default=None, key="tab3", help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible", max_selections=None)
        if len(op1) !=  0:
            graf_linea(df, op1)
    with tab4:
        op2 = st.multiselect('Seleccione variables', options = opcion1, default=opcion1, key="tab4",help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible", max_selections=None)
        if len(op2) !=  0:
            graf_PCA(df, op2)
    
    with tab5:

        st.write('Introduzca las condiciones de busqueda:')
        graf_Theming(df, opcion1)

    with tab6:
        op3 = opcion1
        x5 = st.selectbox('Escoja x: ', op3, key = 'x5')
        x6 = st.selectbox('Escoja y: ', op3, key = 'x6')
        x7 = st.selectbox('Escoja z: ', op3, key = 'x7')

        graf_colorscale(df, x5, x6, x7) 

    with tab7:
        x8 = st.selectbox('Escoja una variable: ', opcion1, key = 'x8')
        graf_barChar(df, x8)
    with tab8:
        op3 = opcion1
        x9 = st.selectbox('Escoja x: ', op3, key = 'x9')
        x10 = st.selectbox('Escoja y: ', op3, key = 'x10')
        x11 = st.selectbox('Escoja z: ', op3, key = 'x11')
        graf_Scatter3D(df, x9, x10, x11)
        
    
elif choice == 'Correlación': 

    st.header('Grado de correlación entre las variables')

    cor_matrix = df.corr().abs()

    tab9, tab10, tab11 = st.tabs(['Tabla de correlación', 'Rango de correlación', 'Mapa de calor'])

    with tab9:
        st.write(cor_matrix)

    with tab10:
        rango = st.slider('Seleccione el rango de correlación deseado', value = [0.5,0.9])
            
        print(rango[0],type(rango[0]))
        st.write(cor_matrix.style.highlight_between(left=rango[0], right=rango[1],inclusive = 'left'))    

    with tab11:
        fig = px.imshow(cor_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig, theme=None, use_container_width=True)

# ==============================================
else:

    pass

# ================================================================================================
import base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('logo_mariposa_2.jpg')

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('logo_mariposa_2.jpg')
# ================================================================================================    


    





    
  



    