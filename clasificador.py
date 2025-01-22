import streamlit as st
from PIL import Image
import os

# Crear un directorio para guardar las imágenes si no existe
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    """Guarda la imagen subida en el directorio UPLOAD_FOLDER."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("Subir y Guardar Imágenes")
    st.write("Sube una imagen y la guardaremos en el servidor.")

    # Widget de subida de archivos
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)

        # Guardar la imagen
        if st.button("Guardar imagen"):
            file_path = save_image(uploaded_file)
            st.success(f"Imagen guardada en: {file_path}")

if __name__ == "__main__":
    main()
