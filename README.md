# Table of Contents

1. [Installation and Libraries](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [How to Run](#run)
5. [Results/Visuals](#results)

## 1. Installation and Libraries <a name="installation"></a>

- sys
- pandas
- sklearn
- sqlalchemy
- re
- nltk
- pickle

## 2. Project Motivation <a name="motivation"></a>

In this project I use disaster messages provided from Figure Eight https://appen.com/ to build a web app that classifies messages using a ML pipeline into 36 categories. 
I approach as follows:
- Build an ETL (Extract, Transform, Load) Pipeline to repair the data.
- Build a supervised learning model using a machine learning Pipeline.
- Build a web app that:
   - takes an input message and gets the classification results of the input in several categories.
   - displays visualisations of the training datasets.
   
## 3. File Descriptions <a name="files"></a>

- Data
   - disaster_categories.csv: CSV file from figure8 for messages
   - disaster_messages.csv: CSV file from figure8 for categories
   - process_data.py: Python script to clean and create a database
   - DisasterResponse.db: database to save clean data to

- Models
   - train_classifier.py: Python script of ML pipeline.
   - classifier.pkl: saved model 

- App
   - template
      - master.html: main page of web app
      - go.html: classification result page of web app
   - run.py: Flask file that runs app

    
## 4. How to Run <a name="run"></a>

1. To run ETL pipeline that cleans data and stores in database:
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2. To run ML pipeline that trains classifier and saves
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3. To run web app cd to app directory and execute $ python run.py
4. To open web app go to http://0.0.0.0:3001/

## 5. Results/Visuals <a name="results"></a>

![image](https://user-images.githubusercontent.com/77011353/119131373-afc8b280-ba39-11eb-86c5-89c035000ac8.png)
![image](https://user-images.githubusercontent.com/77011353/119130688-d0dcd380-ba38-11eb-88c2-25745d6a5b6b.png)
