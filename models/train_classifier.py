import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


# load data from database
def load_data(database_filepath):
    '''
    Function to retreive data from sql database (database_filepath) and split the dataframe into X and y variable
    Input: Databased filepath
    Output: Returns the Features X & target y along with target columns names catgeory_names
    '''

    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('dis_resp_mes',con=engine)
    X = df['message'].values 
    y = df[df.columns[4:]]
    category_names = y.columns.tolist()

    return X, y, category_names

def tokenize(text):
    '''
    Function to clean the text data  and apply tokenize and lemmatizer function
    Return the clean tokens
    Input: text
    Output: cleaned tokenized text as a list object
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    text = word_tokenize(text) 
    text = [w for w in text if w not in stopwords.words("english")]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    return text

def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    Input: N/A
    Output: Returns the model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__n_estimators': [50]}
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='recall_micro', cv=4)
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate a model and return the classificatio and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    y_pred = model.predict(X_test)
  
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))

def save_model(model, model_filepath):
    '''
    Parameters
    model : ML model
        trained and ready to be deployed to production.
    model_filepath : string
        distination to be saved.
    '''
    pickle.dump(model,open(model_filepath,'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
