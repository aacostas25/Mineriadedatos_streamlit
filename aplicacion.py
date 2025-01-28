import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image):
  image = image.convert('L') #convertir a escalas de grises
  image = image.resize((28,28))
  image_array = img_to_array(image) / 255.0
  image_array = np.expand_dims(image_array, axis = 0)
  return image_array
  
def main():
  st.title('Clasificacion de la base de datos Mnist')
  st.markdown('Sube una imagen para clasificar')

uploaded_file = st.file_uploader("Selecciona una imagen: ", type = ["jpg","png","jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption = "imagen subida")

  preprocessed_image = preprocess_image(image)
    
  st.image(preprocessed_image, caption = "imagen subida")
  
if __name__=='__main__':
  main()
  
