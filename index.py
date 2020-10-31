import streamlit as st
import tensorflow as tf
import PIL.Image
import numpy as np
from io import StringIO
from StyleTransfer import StyleTransfer



def resize(image):
    dimension = 512
    width, height = image.size
    maxDim = max(width, height)
    scale = dimension/maxDim
    return image.resize((int(width*scale),int(height*scale)))

if __name__ == "__main__": 
    st.title('Style Transfer')

    st.sidebar.title("Hyperparameters")
    epochs_slider = st.sidebar.slider(
        'Epochs',
        1,10,5
    )

    steps_per_epoch_slider = st.sidebar.slider(
        'Number of Iterations',
        1,200,100
    )
    content_weight_slider = st.sidebar.slider(
        'Peso del contenido',
        0.0,1.0,0.2
    )

    """Vamos a utilizar una red vgg19 como preliminar para aplicar Transfer Learning"""
    styleTransferObject = None
    styleFilePIL = None
    contentFilePIL = None

    #Paso 1 : Seleccionar un estilo

    st.markdown("**Paso 1 :** Seleccionar un estilo")
    #Radio Button
    opciones = ("La noche estrellada","Composicion VII","Subir un estilo propio")
    styleOption = st.radio("¿Cual estilo deseas usar?", opciones)
    imagenMuestra = st.empty()
    if(styleOption == opciones[0]):
        st.write("**La noche estrellada**")
        styleFilePIL = PIL.Image.open('SugestedStyles/LaNocheEstrellada.jpg')
        styleFilePIL = styleFilePIL.convert('RGB')
        imagenMuestra.image(resize(styleFilePIL))
    elif(styleOption == opciones[1]):
        st.write("**Composicion VII**")
        styleFilePIL = PIL.Image.open('SugestedStyles/ComposicionVII.jpg')
        styleFilePIL = styleFilePIL.convert('RGB')
        imagenMuestra.image(resize(styleFilePIL))
    elif(styleOption == opciones[2]):
        styleFile = st.file_uploader("Subir la imagen de estilo propio", type=["png", "jpg"],accept_multiple_files=False)
        if(styleFile):
            styleFilePIL = PIL.Image.open(styleFile)
            styleFilePIL = styleFilePIL.convert('RGB')

    #Paso 2 : Seleccionar un Objetivo

    st.markdown("**Paso 2 :** Seleccionar el objetivo")
    contentFile = st.file_uploader("Subir la imagen objetivo", type=["png", "jpg"],accept_multiple_files=False)
    if(contentFile):
        contentFilePIL = PIL.Image.open(contentFile)
        contentFilePIL = contentFilePIL.convert('RGB')
    if(styleFilePIL is not None and contentFilePIL is not None):
        
        
        st.markdown("**Paso 3 :** Fusionar")
        styleTransferObject = StyleTransfer(styleFilePIL,contentFilePIL,path = False,epochs = epochs_slider,steps_per_epoch = steps_per_epoch_slider, content_weight = content_weight_slider) 
        st.markdown("*Style Image*")
        st.image(styleTransferObject.get_img_style())
        st.markdown("*Content Image*")
        st.image(styleTransferObject.get_img_content())


        if(st.button("Fusionar")):
            styleTransferObject.run()
            st.balloons()

