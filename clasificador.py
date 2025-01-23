import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gzip
import pickle

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
    # Estilos personalizados
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
        }
        .description {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            font-size: 14px;
            color: #888888;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Título y descripción
    st.markdown('<div class="main-title">Clasificación de Imágenes con Modelo Preentrenado</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Sube una imagen y la clasificaremos utilizando un modelo preentrenado.</div>', unsafe_allow_html=True)

    # Widget de subida de archivos
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.subheader("Vista previa de la imagen subida")
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True, output_format="auto")

        # Guardar la imagen
        file_path = save_image(uploaded_file)
        st.success(f"Imagen guardada en: {file_path}")

        # Diccionario de clases
        fashion_mnist_clases = {
            0: "Camiseta/top",
            1: "Pantalón",
            2: "Suéter",
            3: "Vestido",
            4: "Abrigo",
            5: "Sandalia",
            6: "Camisa",
            7: "Zapatilla deportiva",
            8: "Bolso",
            9: "Bota"
        }

        # Botón para clasificar la imagen
        if st.button("Clasificar imagen"):
            with st.spinner("Cargando modelo y clasificando..."):
                model = load_model()
                preprocessed_image = preprocess_image(image)
                prediction = model.predict(preprocessed_image.reshape(1, -1))[0]
                class_name = fashion_mnist_clases.get(prediction, "Clase desconocida")
                st.success(f"La imagen fue clasificada como: {class_name}")

    # Footer
    st.markdown('<div class="footer">© 2025 - Clasificación de imágenes con Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
