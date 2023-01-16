import streamlit as st
import pickle
import numpy as np
import cv2
from tensorflow import keras

# loaded_model = pickle.load(open("model0.sav", "rb"))
loaded_model = pickle.load(open("application/model0.sav", "rb"))

st.title("Fashion MNIST")
st.write("Simply upload a picture of a piece of clothing or take a picture of yourself \
    and we will tell you what it is.")

clothes_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

img_file_buffer = st.camera_input("Take a picture")


if img_file_buffer is not None:
    # to read image file buffer with OpenCV
    bytes_data = img_file_buffer.getvalue()

    # convert to numpy array
    cv2_img = cv2.imdecode(np.frombuffer(
        bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    st.image(cv2_img)

    # Check the type of cv2_img:
    st.write("Original dimension: ", cv2_img.shape)

    # resize image to 28x28
    st.write("Second step: resize image to 28x28")
    cv2_img_2 = cv2.resize(cv2_img, (28, 28))
    st.image(cv2_img_2, width=400)
    st.write("Resized dimension: ", cv2_img_2.shape)

    x_reshape = cv2_img_2.reshape([1, 28, 28, 1])
    st.write("Reshape: ", x_reshape.shape)
    # st.write(type(loaded_model))
    prediction = loaded_model.predict(x_reshape)
    num = np.argmax(prediction)
    st.write("Prediction: ", clothes_dict[num])
