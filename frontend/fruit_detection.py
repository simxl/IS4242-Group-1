import os
import cv2
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import numpy as np


def detect_fruits(image, model, classes):
    # perform preprocessing for image
    image = image.resize((256, 256))
    image = image.convert('RGB')
    image = np.array(image)
    image = image.reshape(-1, image.shape[0] * image.shape[1] * image.shape[2])

    # obtaining fruit predictions
    index = model.predict(image)
    pred = classes[index[0]]
    return pred

if __name__ == '__main__':
    # Get the absolute path of the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Get the absolute path of the parent directory of the current directory
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # Construct the path to the model file
    model_path = os.path.join(parent_dir, "frontend", "model_aug.pkl")

    # model_path = 'model_aug.pkl'

    # loading the Random Forest model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    class_names = ['Apple_Green', 'Apple_Red', 'Banana', 
               'Capsicum_Green', 'Capsicum_Red', 'Capsicum_Yellow', 
               'Lemon', 'Orange', 'Pear', 'Tomato']

    ############################################################################
    # setting up the camera feed
    cam_feed = cv2.VideoCapture(0)

    st.set_page_config(page_title="Fruit Detection")

    st.title("Fruit Detection Application")

    # creating the window to display live video feed
    FRAME_WINDOW = st.image([])
    
    cart_df = pd.DataFrame(columns=["Item"])

    # creating button to detect fruits
    button = st.button('Detect Fruit')

    st.write("## Detected")
    placeholder = st.empty()
    
    while True:
        # obtains frames from camera feed
        ret, frame = cam_feed.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # updating window with the latest frame
        FRAME_WINDOW.image(frame)

        # obtaining image from frame
        img = Image.fromarray(frame)

        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
            break
        
        if button:
            ret, frame = cam_feed.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            # performing image classification
            fruit = detect_fruits(img, model, class_names)

            button = False

            if fruit not in cart_df["Item"].tolist():
                new_cart_df = pd.concat([cart_df, pd.DataFrame.from_records([{"Item": fruit}])], ignore_index=True)

            placeholder.dataframe(new_cart_df, use_container_width=True)
            cart_df = new_cart_df.copy(deep=True)

    cam_feed.release()
    cv2.destroyAllWindows()