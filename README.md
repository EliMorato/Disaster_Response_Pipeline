# Disaster_Response_Pipeline

## Summary
This project is based on using Data Engineering skills to appropriately classify disaster data provided by Figure Eight Inc into categories so each one can be send to the most appropriate disaster relief agency. For this, we need to clean the data, use NLP techniques and finally train a pipeline that fits multilabel data.

The output is a web app which is able to have a message as input and get the most suitable classification. This web app also displays visualizations of the data.

## How to run
To run this project we need to get through four steps:
1. To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
2. To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run the web app:
    `python run.py`
    
4. Go to http://0.0.0.0:3001/

## Detailed content of the repository
This repository contains three folders:
 - **data** is the folder that contains the labelled data (*disaster_messages.csv* and *disaster_categories.csv*) and the python script (*process_data.py*) to process this data and save it in a database (*DisasterResponse.db*).
 
 - **models** is the folder that contains the python script (*train_classifier.py*) to read the processed data from the database (*DisasterResponse.db*), train a model and save it as a pickle file (*classifier.pkl*).
 
 - **app** is the folder that contains the python script (*run.py*) to run the web app and another folder named **templates** with the html files needed to visualize the web app (*master.html* and *go.html*).
 
