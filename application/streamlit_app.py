import streamlit as st
import cv2
import numpy as np
import pickle
from keras.models import load_model

# set page config (must be called as the first Streamlit command)
st.set_page_config(
    page_title="Integrationsseminar",
    page_icon="👕",
)


# import model for deployment
# load model with cache
# @st.cache(allow_output_mutation=True)
# def load_model_path():
#     model_linear = pickle.load(open("application/trained_model.sav", "rb"))
#     model_cnn_1 = load_model("application/cnn_model_1.h5")
#     model_cnn_2 = load_model("application/cnn_model_2.h5")
#     model_cnn_3 = load_model("application/cnn_model_3.h5")
#     return model_linear, model_cnn_1, model_cnn_2, model_cnn_3


# model_linear, model_cnn_1, model_cnn_2, model_cnn_3 = load_model_path()


# import model on local machine (UNCOMMENT BELOW WHEN RUNNING LOCAL)
# model_linear = pickle.load(open("trained_model.sav", "rb"))
# model_cnn_1 = load_model("cnn_model_1.h5")
# model_cnn_2 = load_model("cnn_model_2.h5")
# model_cnn_3 = load_model("cnn_model_3.h5")


# dictionary for categories
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


# linear model prediction
def predict_linear(image):
    # Predict probabilities for each category -> [0.01 0.01 0.2 0. 0.02 0. 0.17 0. 0.59 0. ]
    probs = model_linear.predict_proba(image)[0]
    # Get the index of the highest probability -> 8
    max_prob_index = np.argmax(probs)
    # Get the highest probability -> 0.59
    max_prob = probs[max_prob_index]
    # Get the category name -> Bag
    category = clothes_dict[max_prob_index]
    return category, max_prob


def predict_cnn(model, image):
    # Reshape image to 1x28x28x1
    im_shape = (28, 28, 1)
    image_reshaped = image.reshape(1, *im_shape)
    # Predict the category with the current model
    model_prediction = model.predict(image_reshaped)
    # get the category name
    category_name = clothes_dict[np.argmax(model_prediction)]
    # get the highest value
    prob = np.max(model_prediction)

    return category_name, prob


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
    cv2_img_resized = cv2.resize(cv2_img, (28, 28))
    st.image(cv2_img_resized, width=400)
    st.write("Resized dimension: ", cv2_img_resized.shape)

    # image as numpy array
    st.write("The image as numpy array")
    st.write("Original shape: ", cv2_img.shape)
    st.write("Resized shape: ", cv2_img_resized.shape)
    st.write(cv2_img_resized)

    # flatten image
    img_flatten = cv2_img_resized.reshape(1, -1)
    st.write("Flatten image: ", img_flatten.shape)
    st.write(img_flatten)
    st.write("Rescale values by dividing by 255: ")
    img_flatten_rescaled = img_flatten / 255
    st.write(img_flatten_rescaled)

    # predict category linear
    category_linear, category_linear_prob = predict_linear(
        img_flatten_rescaled)

    # predict category cnn
    category_cnn_1, category_cnn_prob_1 = predict_cnn(
        model_cnn_1, cv2_img_resized)
    category_cnn_2, category_cnn_prob_2 = predict_cnn(
        model_cnn_2, cv2_img_resized)
    category_cnn_3, category_cnn_prob_3 = predict_cnn(
        model_cnn_3, cv2_img_resized)

    st.write("The model predicts the following categories:")
    st.write(
        f"Linear: {category_linear} - probability: {category_linear_prob}")
    st.write(
        f"1 CNN layer: {category_cnn_1} - probability: {category_cnn_prob_1}")
    st.write(
        f"2 CNN layer: {category_cnn_2} - probability: {category_cnn_prob_2}")
    st.write(
        f"3 CNN layer: {category_cnn_3} - probability: {category_cnn_prob_3}")
