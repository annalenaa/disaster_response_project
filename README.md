# Disaster Reponse Pipeline

## Description

This project is part of the Udacity Data Scientist Nanodegree. The aim of the project is to practice some skills in building an ETL and a machine learning pipeline. The goal is to build a model that classifies disaster messages and deploy it in a web app where users can type in a message and get the classification result for it. 

In the first part of the project the data was preprocessed and prepared for the following analysis and modeling. 

In the second part the messages were processed through a NLP pipeline in order to feed them to a machine learning alhorithm. The resulting model (trained using cross validation) was then stored as a pickle file and is being used for making predictions on new messages.

## Libraries

- pandas
- numpy
- sqlalchemy
- scikit-learn
- nltk (download 'punkt','wordnet',and 'averaged_perceptron_tagger')
- flask
- plotly
- json

## Content

**1. app:** folder containing the HTML and python scripts for the web app
    - *run.py* (script that runs the flask web app)
    - *templates* (folder containing the html files for the app)
    
**2. data:** 
    - *disaster_categories.csv* (original category data)
    - *diaster_messages.csv* (original messages)
    - *DisasterResponse.db* (database containing the processed data that is a combination of the 2 original data sets)
    - *process_data.py* (python script that does the data preprocessing and saves the new, combined data in the database)
    
**3. models:** 
    - *train_classifier.py* (python script that loads the prepared data from the database, prepares it for modeling and trains a classifier)
    - *classifier.pkl* (trained classifier as a pickle file)
    
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements

The data for this project was kindly provided by Figure Eight. The majority of the code for the app as well as some starter code for the other scripts were provided by Udacity. 


