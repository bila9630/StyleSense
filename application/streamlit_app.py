import streamlit as st
import random
import cv2
import numpy as np
import pickle

st.set_page_config(
    page_title="Integrationsseminar",
    page_icon="👕",
)

# loading the saved model
# uncomment the following line if you are running the app locally
#loaded_model = pickle.load(open("trained_model.sav", "rb"))
loaded_model = pickle.load(open("application/trained_model.sav", "rb"))

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

def model_prediction(image):
    category = loaded_model.predict(image)
    category_name = clothes_dict[category[0]]
    return category_name

def random_class(image):
    # generate random number between 0 and 9
    random_number = random.randint(0, 9)

    category = clothes_dict[random_number]
    confidence = round(random.random(), 2)
    return category, confidence


st.title("Fashion MNIST")
st.write("Simply upload a picture of a piece of clothing or take a picture of yourself \
    and we will tell you what it is.")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # to read image file buffer with OpenCV
    bytes_data = img_file_buffer.getvalue()

    # convert to numpy array and read as grayscale
    st.write("First step: convert to numpy array and read as grayscale")
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

    # image as numpy array
    st.write("The image as numpy array")
    st.write("Original shape: ", cv2_img.shape)
    st.write("Image shape: ", cv2_img_2.shape)
    st.write(cv2_img_2)

    # flatten image
    img_flatten = cv2_img_2.reshape(1, -1)
    st.write("Flatten image: ", img_flatten.shape)
    st.write(img_flatten)
    st.write("Rescale values by dividing by 255: ")
    img_flatten_2 = img_flatten / 255
    st.write(img_flatten_2)

    # category, confidence = random_class(img_flatten_2)
    # st.write(f"Category: {category}")
    # st.write(f"Confidence: {confidence}")

    category2 = model_prediction(img_flatten_2)
    st.write(f"Category: {category2}")
