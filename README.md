  Table of Contents
  
  1. Istallation
  2. Project overview
  3. File description
  4. How to run
  5. Results/Visuals
   
   
   
   
   
   - To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
