import streamlit as st
import cv2
import numpy as np
import pickle

# set page config (must be called as the first Streamlit command)
st.set_page_config(
    page_title="Integrationsseminar",
    page_icon="👕",
)


# import model for deployment
# load model with cache
@st.cache(allow_output_mutation=True)
def load_model_path():
    model_linear = pickle.load(open("application/trained_model.sav", "rb"))
    model_1_layer = pickle.load(open("application/model_1_layer.sav", "rb"))
    model_2_layer = pickle.load(open("application/model_2_layer.sav", "rb"))
    model_3_layer = pickle.load(open("application/model_3_layer.sav", "rb"))
    return model_linear, model_1_layer, model_2_layer, model_3_layer

model_linear, model_1_layer, model_2_layer, model_3_layer = load_model_path()


# import model on local machine (UNCOMMENT BELOW WHEN RUNNING LOCAL)
# model_linear = pickle.load(open("trained_model.sav", "rb"))
# model_1_layer = pickle.load(open("model_1_layer.sav", "rb"))
# model_2_layer = pickle.load(open("model_2_layer.sav", "rb"))
# model_3_layer = pickle.load(open("model_3_layer.sav", "rb"))

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


# Create a list to store the model
cnn_models = [model_1_layer, model_2_layer, model_3_layer]


def predict_nn(image):
    # Reshape image to 1x28x28x1
    image_reshape_nn = image.reshape(1, 28, 28, 1)

    # Create a dictionary to store the predictions of each model
    predictions = {}
    # Iterate over the models
    for i, model in enumerate(cnn_models):
        # Predict the category with the current model
        model_prediction = model.predict(image_reshape_nn)
        # get the index where the highest value is
        category = np.argmax(model_prediction)
        # get the highest value
        prob = np.max(model_prediction)
        # Add the prediction to the dictionary
        predictions["model_{}_category".format(i + 1)] = clothes_dict[category]
        predictions["model_{}_prob".format(i + 1)] = prob

    return predictions


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

    # predict category neural network
    predictions_nn = predict_nn(cv2_img_resized)

    st.write(
        f"Category linear: {category_linear} with probability: {category_linear_prob}")
    st.write(
        f"Category 1 CNN layer: {predictions_nn['model_1_category']} with probability: {predictions_nn['model_1_prob']}")
    st.write(
        f"Category 2 CNN layer: {predictions_nn['model_2_category']} with probability: {predictions_nn['model_2_prob']}")
    st.write(
        f"Category 3 CNN layer: {predictions_nn['model_3_category']} with probability: {predictions_nn['model_3_prob']}")
