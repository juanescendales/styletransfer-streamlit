#https://www.tensorflow.org/tutorials/generative/style_transfer
import tensorflow as tf
from Model import StyleContentModel
import numpy as np
import PIL.Image
import streamlit as st

class StyleTransfer:
    # IO Functions
    def randomImage(self,imagen):
        width, height = imagen.size
        arr = np.random.randint(0,255,(height,width,3), dtype='uint8')
        return PIL.Image.fromarray(arr)

    def get_img_content(self):
        return self.tensor_to_image(self.content_image)

    def get_img_style(self):
        return self.tensor_to_image(self.style_image)

    def get_stylized_image(self):
        return self.tensor_to_image(self.stylized_image)

    def tensor_to_image(self,tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def decode_image(self,img):
        max_dim = 512
        img = tf.keras.preprocessing.image.img_to_array(img)/255
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    
    def __init__(self,style, content,epochs=10,steps_per_epoch = 100,content_weight = 0.2):
        #Default Hyper parameters
        #self.style_weight= style_weight                    
        self.content_weight= content_weight                    
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        #Images
        self.content_image = self.decode_image(content)
        self.style_image = self.decode_image(style)

        self.stylized_image = tf.Variable(self.content_image) # Si se quiere probar con una imagen random : tf.Variable(self.decode_image(self.randomImage(content)))
        #Layers a usar de la VGG19
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


    #Run Model
    def run(self):
        """Una red convolucional mas alla de sus funciones de clasificacion , es un extractor de componentes desde simples a complejas
        por lo tanto en ciertas capas podemos tener acceso a ciertas caracteristicas de la imagen donde podemos ingresar un estilo, ya 
        que en cada capa de la red esta va extrayendo caracteristicas cada vez mas complejas."""
        modelo = StyleContentModel(self.style_layers, self.content_layers,self.content_weight)
        modelo.defineTargets(self.style_image,self.content_image)
        #Streamlit UI
        st.markdown("*Result Image*")
        progressImage = st.empty()
        latest_iteration = st.empty()
        bar = st.progress(0.0)
        latest_iteration.text("No Epoch")

        #Model Execution
        for epoch in range(1,self.epochs+1):
            progressImage.image(self.get_stylized_image())
            for _ in range(self.steps_per_epoch):
                modelo.train_step(self.stylized_image)
            latest_iteration.text(f'Epoch {epoch}')
            bar.progress(epoch/self.epochs)