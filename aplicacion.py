import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn

def preprocess_image(image):
  image = image.convert('L') #convertir a escalas de grises
  image = image.resize((28,28))
  image_array = img_to_array(image) / 255.0
  image_array = np.expand_dims(image_array, axis = 0)
  return image_array

def load_model():
  filename = "modelo_entrenado.pkl.gz"
  with gzip.open(filename, 'rb') as f:
    model = pickle.load(f)
  return model

def main():
  st.title('Clasificacion de la base de datos Mnist')
  st.markdown('Sube una imagen para clasificar')

uploaded_file = st.file_uploader("Selecciona una imagen: ", type = ["jpg","png","jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption = "imagen subida")

  preprocessed_image = preprocess_image(image)
    
  st.image(preprocessed_image, caption = "imagen subida")


if st.button("Clasificar imagen"):
      st.markdown("Imagen clasificada")
      model = load_model()
      # Necesitamos un arreglo de (1,784) por lo cual le vamos a hacer un reshape
      prediction = model.predict(preprocessed_image.reshape(1,-1))
      st.markdown(f"La imagen fue clasificada como: {prediction}")
  
if __name__=='__main__':
  main()
  
