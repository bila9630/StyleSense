import streamlit as st
import cv2
import numpy as np
import pickle
from keras.models import load_model
import xgboost as xgb

# set page config (must be called as the first Streamlit command)
st.set_page_config(
    page_title="Integrationsseminar",
    page_icon="👕",
)


# import model for deployment
# load model with cache
@st.cache_resource
def load_model_path():
    model_decision_tree = pickle.load(
        open("application/decision_tree.pkl", "rb"))
    model_cnn_1 = load_model("application/cnn_model_1.h5")
    model_cnn_2 = load_model("application/cnn_model_2.h5")
    model_cnn_3 = load_model("application/cnn_model_3.h5")
    model_rf = pickle.load(open("application/model_rf.pkl", "rb"))
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model("application/model_xgb.txt")
    return model_decision_tree, model_cnn_1, model_cnn_2, model_cnn_3, model_rf, model_xgb


model_decision_tree, model_cnn_1, model_cnn_2, model_cnn_3, model_rf, model_xgb = load_model_path()


# import model on local machine (UNCOMMENT BELOW WHEN RUNNING LOCAL)
# model_decision_tree = pickle.load(open("decision_tree.pkl", "rb"))
# model_rf = pickle.load(open("model_rf.pkl", "rb"))
# model_cnn_1 = load_model("cnn_model_1.h5")
# model_cnn_2 = load_model("cnn_model_2.h5")
# model_cnn_3 = load_model("cnn_model_3.h5")

# model_xgb = xgb.XGBClassifier()
# model_xgb.load_model("model_xgb.txt")

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
def predict_linear(model, image):
    # Predict probabilities for each category -> [0.01 0.01 0.2 0. 0.02 0. 0.17 0. 0.59 0. ]
    probs = model.predict_proba(image)[0]

    # Get the index of the highest probability -> 8
    max_prob_index = np.argmax(probs)

    # Get the highest probability -> 0.59
    max_prob = probs[max_prob_index]

    # Get the category name -> Bag
    category_name = clothes_dict[np.argmax(probs)]

    return category_name, max_prob


def predict_xgboost(model, image):
    # Predict the class label
    y_pred = model.predict(image)
    # Get the index of the predicted class
    class_index = int(y_pred[0])
    # Get the class name
    class_name = clothes_dict[class_index]
    # Get the predicted class probability
    class_proba = model.predict_proba(image)[0][class_index]
    return class_name, class_proba


def predict_cnn(model, image):
    # Rescale values by dividing by 255
    image_rescaled = image / 255

    # Reshape image to 1x28x28x1
    im_shape = (28, 28, 1)
    image_reshaped = image_rescaled.reshape(1, *im_shape)

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
    st.write("Rescale values by dividing by 255 (optional): ")
    img_flatten_rescaled = img_flatten / 255
    st.write(img_flatten_rescaled)

    # predict category decision tree classifier
    category_decision_tree, category_decision_tree_prob = predict_linear(
        model_decision_tree, img_flatten)
    category_random_forest, category_random_forest_prob = predict_linear(
        model_rf, img_flatten)
    category_xgb, category_xgb_prob = predict_xgboost(
        model_xgb, img_flatten)

    # predict category cnn
    category_cnn_1, category_cnn_prob_1 = predict_cnn(
        model_cnn_1, cv2_img_resized)
    category_cnn_2, category_cnn_prob_2 = predict_cnn(
        model_cnn_2, cv2_img_resized)
    category_cnn_3, category_cnn_prob_3 = predict_cnn(
        model_cnn_3, cv2_img_resized)

    st.write("The model predicts the following categories:")
    st.write(
        f"Decision Tree: {category_decision_tree} - probability: {category_decision_tree_prob:.2f}")
    st.write(
        f"XGBoost: {category_xgb} - probability: {category_xgb_prob:.2f}"
    )
    st.write(
        f"Random Forest: {category_random_forest} - probability: {category_random_forest_prob:.2f}")
    st.write(
        f"1 CNN layer: {category_cnn_1} - probability: {category_cnn_prob_1:.2f}")
    st.write(
        f"2 CNN layer: {category_cnn_2} - probability: {category_cnn_prob_2:.2f}")
    st.write(
        f"3 CNN layer: {category_cnn_3} - probability: {category_cnn_prob_3:.2f}")
