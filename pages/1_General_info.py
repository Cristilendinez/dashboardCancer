"""
General information about the mission. 
By default, all telescopes are selected.
For each one, it charges the image with the logo, and then read the text written in the 
utils/general_description dictionary.
The dictionary is formatted with sub dictionary:
For each telescope name, you have a key for the Instruments and the Surveys. This way, we can
loop through the information, have a standardized format and an easier readability of the dictionary.

"""


import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from PIL import Image
import base64
import streamlit as st

#from telescopes.main_info import info
#from utils.general_description import description
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
#st.write("logo cancer mariposa = https://www.freepik.com/premium-vector/breast-cancer-awareness-month-with-butterfly-sign-pink-ribbons-vector-illustration-design-poster_32547604.htm")


# Imagen
#st.image('https://fotografias.antena3.com/clipping/cmsimages02/2019/10/17/1C516914-50F2-433C-96EE-FDCBC10D4638/97.jpg?crop=1920,1080,x0,y177&width=1600&height=900&optimize=low&format=webply')

st.title("Cancer de mama 	👱‍♀️")
st.caption(" **El cáncer de mama es el cáncer más común diagnosticado en mujeres y representa más de 1 de cada 10 nuevos diagnósticos de cáncer cada año. Es la segunda causa más común de muerte por cáncer entre las mujeres en el mundo. Anatómicamente, el seno tiene glándulas productoras de leche frente a la pared torácica. Se encuentran en el músculo pectoral mayor, y hay ligamentos que sostienen el seno y lo unen a la pared torácica. Quince a 20 lóbulos dispuestos circularmente para formar el pecho. La grasa que cubre los lóbulos determina el tamaño y la forma de los senos. Cada lóbulo está formado por lobulillos que contienen las glándulas responsables de la producción de leche en respuesta a la estimulación hormonal. El cáncer de mama siempre evoluciona en silencio. La mayoría de los pacientes descubren su enfermedad durante su examen de rutina. Otros pueden presentar un bulto en el seno descubierto accidentalmente, cambio en la forma o el tamaño de los senos, o secreción del pezón. Sin embargo, la mastalgia no es infrecuente. Se debe realizar un examen físico, imágenes, especialmente una mamografía, y una biopsia de tejido para diagnosticar el cáncer de mama. La tasa de supervivencia mejora con el diagnóstico precoz. El tumor tiende a diseminarse por vía linfática y hematológica, dando lugar a metástasis a distancia y mal pronóstico. Esto explica y enfatiza la importancia de los programas de detección del cáncer de mama.** ")

st.markdown("[Cancer de Mama](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))")
st.markdown("[Centro para el control de enfermedades](https://www.cdc.gov/spanish/cancer/breast/basic_info/treatment.html)")

st.subheader("Factores de riesgo del cáncer de mama 🎗️")
st.caption("- Edad: El riesgo de cáncer de mama aumenta con la edad. La mayoría de los cánceres de mama se diagnostica después de los 50 años de edad.")
st.caption("- Mutaciones genéticas. Las mujeres que han heredado cambios (mutaciones) heredados en ciertos genes, tales como en el BRCA1 y el BRCA2, tienen mayor riesgo de presentar cáncer de mama y de ovario.")
st.caption("- Historial reproductivo. El comienzo de la menstruación antes de los 12 años de edad y de la menopausia después de los 55 años de edad exponen a las mujeres a hormonas por más tiempo, lo cual aumenta el riesgo de cáncer de mama.")
st.caption("- Tener mamas densas. Las mamas densas mamas densas tienen más tejido conjuntivo que tejido adiposo, lo cual, a veces, puede hacer difícil la detección de tumores en una mamografía.")
st.caption("- Antecedentes personales de cáncer de mama o ciertas enfermedades de las mamas que no son cancerosas. Las mujeres que han tenido cáncer de mama tienen mayores probabilidades de tener esta enfermedad por segunda vez.")
st.caption("- Antecedentes familiares de cáncer de mama o cáncer de ovario. El riesgo de una mujer de tener cáncer de mama es mayor si su madre, una hermana o una hija (parientes de primer grado) o varios integrantes de la familia por el lado paterno o materno han tenido cáncer de mama o cáncer de ovario.")
st.caption("- Tratamientos previos con radioterapia. Las mujeres que han recibido radioterapia en el pecho o las mamas antes de los 30 años de edad ")
st.caption("- Exposición al medicamento dietilestilbestrol. Las mujeres que tomaron dietilestilbestrol, o cuyas madres tomaron dietilestilbestrol cuando estaban embarazadas de ellas, tienen un mayor riesgo de tener cáncer de mama.")

st.subheader("Tratamientos 🌌")
st.caption("- Cirugía: Una operación en la que los médicos cortan el tejido con cáncer.")
st.caption("- Quimioterapia: Se usan medicamentos especiales para reducir o matar las células cancerosas.")
st.caption("-Terapia hormonal: Impide que las células cancerosas obtengan las hormonas que necesitan para crecer.")
st.caption("- Inmunoterapia: Trabaja con el sistema inmunitario de su cuerpo para ayudarlo a combatir las células cancerosas o a controlar los efectos secundarios que causan otros tratamientos contra el cáncer.")
st.caption("- Radioterapia: Se usan rayos de alta energía (similares a los rayos X) para matar las células cancerosas.")

#Imagen
st.image('hospital.jpg')
st.write("imagen hospital= https://www.elnorte.com/aplicacioneslibre/preacceso/articulo/default.aspx?__rval=1&urlredirect=https://www.elnorte.com/hospital-seguro-checa-su-higiene/ar2189732?referer=--7d616165662f3a3a6262623b727a7a7279703b767a783a--")

st.subheader("Tipos de tumores mamarios 🙈")

st.sidebar.write("Carcinoma ductal invasivo (CDI)")
st.sidebar.write("Carcinoma inflamatorio de mama")
st.sidebar.write("Carcinoma lobulillar “in situ” (CLIS)")
st.sidebar.write("La enfermedad de Paget de la mama")
#radio= st.sidebar.radio( "Select telescopes to disp",["CIM","CLIS", "EPM", "TFCF"])


tab1, tab2, tab3, tab4= st.tabs(["Carcinoma ductal invasivo(CDI)","Carcinoma inflamatorio de mama", "Carcinoma lobulillar “in situ” (CLIS)", "La enfermedad de Paget de la mama"])

with tab1:
    st.subheader("Carcinoma ductal invasivo(CDI)")
    st.write("Es el tipo más común de cáncer de mama, suponiendo aproximadamente el ochenta por ciento de los casos. Se desarrolla a partir de células de origen epitelial (carcinoma),"
    "Al ser un tumor invasivo, tiene la capacidad para diseminarse hacia los ganglios linfáticos y otras zonas del cuerpo."
    "Es el tipo más frecuente de cáncer de mama en los hombres.")
    
    # Imagen
    st.image('carcinoma_ductal.jpg')
    st.write("carcinoma ductal = https://es.wikipedia.org/wiki/Carcinoma_ductal_%27in_situ%27#/media/Archivo:Breast_DCIS_histopathology_(1).jpg")

with tab2:
    st.subheader("Carcinoma inflamatorio de mama")
    # Texto
    st.write("Es un tipo de cáncer de mama con un comportamiento biológicamente agresivo, pero poco frecuente (menos del 3 por ciento de casos diagnosticados)."
    "Es un tumor de crecimiento rápido, por lo que resulta primordial reconocer estos síntomas para confirmar el diagnóstico e iniciar un tratamiento adecuado lo antes posible."
    "En ocasiones se puede confundir con un proceso infeccioso denominado mastitis.")
    
    #Imagen
    st.image('carcinoma_inflamatorio.jpg')
    st.write("carcinoma inflamatorio = https://es.wikipedia.org/wiki/Carcinoma_ductal_%27in_situ%27#/media/Archivo:Breast_DCIS_histopathology_(1).jpg")

with tab3:
    st.subheader("Carcinoma lobulillar “in situ” (CLIS)")
    #Texto
    st.write("se debe al crecimiento celular anómalo de una o varias áreas del lobulillo. Su presencia indica que existe un mayor riesgo de que esa persona desarrolle un cáncer de mama invasivo más adelante,"
    "que puede o no desarrollarse a partir de las zonas originales del carcinoma lobulillar in situ. El carcinoma lobulillar in situ se diagnostica generalmente antes de la menopausia, más frecuentemente entre los 40 y 50 años de edad.")

    # Imagen
    st.image('carcinoma_lobular.jpg')
    st.write("carcinoma lobulillar= https://www.google.com/search?q=carcinoma+lobulillar&rlz=1C1UEAD_esES1035ES1035&sxsrf=AJOqlzV87tM-VC7CRD5eP2Za-B7fOQBFvw:1674044353679&source=lnms&tbm=isch&sa=X&ved=2ahUKEwix9qeYjdH8AhWkcKQEHf6wAJMQ_AUoAXoECAEQAw&biw=1536&bih=664&dpr=1.25#imgrc=bsnVpu7fR3oTyM")
    pass
with tab4:
    st.subheader("La enfermedad de Paget de la mama")
    st.write("Enfermedad de Paget mamaria aparece con más frecuencia después de los 50 años. La mayoría de las personas con este diagnóstico también tienen cáncer mamario ductal subyacente, ya sea in situ (en el lugar inicial) o, menos común, cáncer mamario invasivo. Pocas veces, la enfermedad de Paget mamaria se encuentra solo en el pezón.")
    # Inmagen
    st.image('mamaria_paguet.jpg')
    st.write("enfermedad de piaget de la mama= https://www.google.com/search?q=La+enfermedad+de+Paget+de+la+mama+histologia&tbm=isch&ved=2ahUKEwiC36q1jdH8AhWooicCHQ1-Ag0Q2-cCegQIABAA&oq=La+enfermedad+de+Paget+de+la+mama+histologia&gs_lcp=CgNpbWcQAzoECCMQJzoECAAQHlCEBVijFWC9F2gAcAB4AIABvgGIAY0NkgEEMC4xMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=_uPHY4LoIKjFnsEPjfyJaA&bih=664&biw=1536&rlz=1C1UEAD_esES1035ES1035#imgrc=LmKQDPp9ChlPCM")
    pass


st.subheader("Informacion de las imagenes de los logos")
#st.write("carcinoma lobular= https://es.wikipedia.org/wiki/Carcinoma_lobulillar_%27in_situ%27")
#st.write("carcinoma inflamatorio = https://es.wikipedia.org/wiki/Carcinoma_ductal_%27in_situ%27#/media/Archivo:Breast_DCIS_histopathology_(1).jpg")
#st.write("carcinoma lobulillar= https://www.google.com/search?q=carcinoma+lobulillar&rlz=1C1UEAD_esES1035ES1035&sxsrf=AJOqlzV87tM-VC7CRD5eP2Za-B7fOQBFvw:1674044353679&source=lnms&tbm=isch&sa=X&ved=2ahUKEwix9qeYjdH8AhWkcKQEHf6wAJMQ_AUoAXoECAEQAw&biw=1536&bih=664&dpr=1.25#imgrc=bsnVpu7fR3oTyM")
#st.write("enfermedad de piaget de la mama= https://www.google.com/search?q=La+enfermedad+de+Paget+de+la+mama+histologia&tbm=isch&ved=2ahUKEwiC36q1jdH8AhWooicCHQ1-Ag0Q2-cCegQIABAA&oq=La+enfermedad+de+Paget+de+la+mama+histologia&gs_lcp=CgNpbWcQAzoECCMQJzoECAAQHlCEBVijFWC9F2gAcAB4AIABvgGIAY0NkgEEMC4xMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=_uPHY4LoIKjFnsEPjfyJaA&bih=664&biw=1536&rlz=1C1UEAD_esES1035ES1035#imgrc=LmKQDPp9ChlPCM")
st.write("logo cancer mundo= https://www.sysmex.co.uk/products/life-science/breast-cancer.html")
st.write("logo cancer mariposa = https://www.freepik.com/premium-vector/breast-cancer-awareness-month-with-butterfly-sign-pink-ribbons-vector-illustration-design-poster_32547604.htm")
#st.write("imagen hospital= https://www.elnorte.com/aplicacioneslibre/preacceso/articulo/default.aspx?__rval=1&urlredirect=https://www.elnorte.com/hospital-seguro-checa-su-higiene/ar2189732?referer=--7d616165662f3a3a6262623b727a7a7279703b767a783a--")    



