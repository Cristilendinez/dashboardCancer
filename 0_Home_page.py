"""

This is the main page, that you have to run with "streamlit run" to launch the app locally.
Streamlit automatically create the tabs in the left sidebar from the .py files located in /pages
Here we just have the home page, with a short description of the tabs, and some images

"""

import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.image as mpimg

st.set_page_config(
    page_title="Home page",
    page_icon="👋",
    layout="wide")

st.image('Cancer_mama.png')

# Main Description
st.markdown("## 👋 Bienvenido al Análisis de Predicción sobre el Cancer de Mama!")
st.markdown("Diseñado por Cristina Lendinez y Maria Fernanda Mendoza (https://github.com/menpett)")
st.markdown("Esta aplicacion se encuentra en continua mejora, si tienes alguna duda consultanos")
st.markdown("Si quieres conocer otros ejemplos de otras categoprias diferentes pincha aqui (https://streamlit.io/gallery?category=streamlit-templates)")


# Description of the features. 
st.markdown(
    """
    ### Select on the left panel what you want to explore:

    - 🔭  General info, expondremos informacion sobre el cancer de mama y sus variables
    
    - 🎨  Análisis Exploratorio de Datos, expondremos una breve vision de nuestros datos, su extension y un analisis estadistico básico

    -  🌌 Machine Learning, expondremos nuestros datos a los diferentes algoritmos de prediccion.

    - ✨  Prediccion, sveremos si nuestros algoritmos son capaces de clasificar de forma correcta.

    

    \n  
 
    """
)

# ================================================================================================
# Mission logos
st.image('mujer.png')

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

set_background('logo_mundo.jpeg')

















