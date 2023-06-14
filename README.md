# Integrationsseminar
Check out our application: [https://bila9630-integrationsseminar-applicationstreamlit-app-a1fjrt.streamlit.app/](https://bila9630-stylesense-applicationstreamlit-app-okes4r.streamlit.app/)


## Project description

Idea: Developing an application that utilizes CNN to identify clothing items worn by the user
<br>We use FASHION-MNIST as our dataset. It consists of 70,000 images of clothing items. The images are 28x28 pixels and are grayscale. The dataset is split into 60,000 training images and 10,000 test images. The images are divided into 10 classes, which are: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. The dataset is available on Kaggle: https://www.kaggle.com/zalando-research/fashionmnist
<br>We trained multiple models like decision tree, random forest, xgboost and multiple variance of Convolutional Neural Network (CNN). The best model was a CNN with 3 convolutional layers and 2 dense layers. The model achieved an accuracy of 91.5% on the test set.
<br>Group: Hannah Schult, Sofie Pischl, Viet Duc Kieu

## Navigation
- **Model Development** can be found in the folder [/analytics](/analytics)
- **Application** can be found in the folder [/application](/application)
- **Data** can be found in the folder [/data](/data)
- **Presentation** can be found in the folder [/docs](/docs/FashionMNIST%20-%20Integrationsseminar.pptx)
- **Dokumentation** can be found in the folder [/docs](/docs/Seminararbeit.pdf)

## application
### how to start locally
```
cd application
pip install -r requirements.txt
streamlit run streamlit_app.py
```
application is now running on http://localhost:8501

to freeze the requirements:
```
pip freeze > requirements.txt
```

to create a virtual environment:
```
# create a virtual environment
virtualenv env
# activate the virtual environment
env\Scripts\activate
```

## Trouble shooting
When the streamlit app is not running locally:
- adjust the path to the model in the streamlit_app.py file (there are comment that you just need to uncomment in the file)

## data
source: https://www.kaggle.com/datasets/zalando-research/fashionmnist
