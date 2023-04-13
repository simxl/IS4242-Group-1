# IS4242-Group-1

## Dependencies
To install the required dependencies, run the command: 

```
pip install -r requirements.txt
```

Use Google Colab to run the notebooks in the model folder. Additional dependencies have been included in each notebook.

## `model` folder
The model folder contains the notebooks used for data preprocessing/augmentation and model development (SVM, RF, CNN). The data folder contains the datasets used to train the models and conduct our experiments.

## `demo application` folder
The demo application folder contains the frontend streamlit script and the trained random forest model used for the fruit image classification. 

To start the application, run the following command:

```
streamlit run frontend/fruit_detection.py
```