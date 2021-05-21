### Table of Contents

1. [Installation and Libraries](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [How to Run](#run)
5. [Results/Visuals](#results)

## Installation and Libraries  <a name="installation"></a>

- sys
- pandas
- sklearn
- sqlalchemy
- re
- nltk
- pickle

## Project Motivation<a name="motivation"></a>

In this project, I analyze disaster messages provided from Figure Eight https://appen.com/ and build a web app that classifies messages using a ML pipeline into 36 categories.   
   
   
## File Descriptions <a name="files"></a>

- data
|- disaster_categories.csv  # CSV file from figure8 for messages 
|- disaster_messages.csv    # CSV file from figure8 for categories
|- process_data.py          # Python script to clean and create a database
|- DisasterResponse.db      # database to save clean data to

- models
|- train_classifier.py  # Python script of ML pipeline.
|- classifier.pkl       # saved model 

- app
| - template
| |- master.html  # main page of web app
| |- go.html      # classification result page of web app
|- run.py         # Flask file that runs app


## How to Run <a name="run"></a>

1. To run ETL pipeline that cleans data and stores in database
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2. To run ML pipeline that trains classifier and saves
     python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3. To run web app cd to app directory and execute $ python run.py
4. To open web app go to http://0.0.0.0:3001/
