# Data Engineering Disaster Response


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Acknowledgements](#acknowledgements)


## Introduction <a name="introduction"></a>

This is an attempt to buld an ETL and ML pipelines as well as a simple web app to use the created model to classify messages in case of disasters.


## Installation <a name="installation"></a>

The following packages is needed for the scripts and applications to work:
- nltk
- re
- numpy
- pandas
- sklearn
- sqlalchemy
- pickle


## Usage <a name="usage"></a>

1. Run the following commands in the project's root directory to set up your database and model (This is not nescessary, as the DisasterResponse.db and classifier.pkl are both included already, but you can rund the commands to validate the code).
    
    - To run ETL pipeline that cleans data and stores in database
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves
        'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
      
2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click http://0.0.0.0:3001 to open the homepage (please note that you might need to use a port other than 3001 depending on your machine configuration)

Node: In case you want to use the models/classifier.pkl directly please extract it first as it is currently compressed in .zip format 


## Acknowledgements <a name="acknowledgements"></a>

[Udacity](https://learn.udacity.com/) for provding the training and scripts templates.
[Appen](https://www.appen.com/) for providing the datasets used to train the model.


