import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import gzip

# Crear un directorio para guardar las imágenes si no existe
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    """Guarda la imagen subida en el directorio UPLOAD_FOLDER."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    # model = tf.keras.models.load_model('modelito.keras')
    # Cargar el modelo comprimido
    filename = 'model_trained2.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    """Preprocesa la imagen para que sea compatible con el modelo."""
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar los píxeles
    image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión batch
    return image_array

def main():
    st.title("Clasificación de Imágenes con Modelo Preentrenado")
    st.write("Sube una imagen y la clasificaremos utilizando un modelo preentrenado.")

    # Widget de subida de archivos
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)
        st.image(preprocessed_image, caption="Imagen subida", use_container_width=True)

        # Guardar la imagen
        file_path = save_image(uploaded_file)
        st.success(f"Imagen guardada en: {file_path}")

        # Clasificar la imagen
        if st.button("Clasificar imagen"):
            model = load_model()
            preprocessed_image = preprocess_image(image)
            # prediction = model.predict(image)
            prediction = model.predict(preprocessed_image)
            # predicted_class = np.argmax(prediction, axis=1)[0]
            st.write(f"genial, modelo entrenado")  # Para ver los resultados detallados
            st.write(f"La imagen fue clasificada como la clase: {prediction}")


if __name__ == "__main__":
    main()
