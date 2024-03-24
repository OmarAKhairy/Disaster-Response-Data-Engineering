# Data Engineering Disaster Response


## Table of Contents

1. [Project Summary](#summary)
2. [Dependencies](#dependencies)
3. [Project Structure](#structure)
4. [Usage](#usage)
5. [Acknowledgements](#acknowledgements)


## Project Summary <a name="summary"></a>

Disaster Response Classifier Web Application

In times of crisis and disaster, effective communication and swift action are crucial for saving lives and mitigating damage. 
The Disaster Response Classifier is a web application that leverages machine learning to categorize incoming messages, enabling responders to identify and prioritize urgent needs more efficiently.

### Key Features:
- Message Classification: Utilizes a trained machine learning model to classify incoming messages into predefined categories such as "aid_related" "medical_help" "search_and_rescue" and more.
- Actionable Insights: Empowers emergency response teams with actionable insights by highlighting critical messages and trends.
- Visualization: Results are visualized in an easy-to-understand format, allowing responders to prioritize actions effectively.

### Why It Matters:
Efficiency: Automates the categorization process, saving valuable time and resources for emergency response teams.
Effectiveness: Ensures that critical messages are identified promptly, leading to faster and more targeted response efforts.
Scalability: Can be easily scaled to handle large volumes of messages during major disasters or crises.


## Dependencies <a name="dependencies"></a>

The following packages is needed for the scripts and applications to work:
- nltk
- re
- numpy
- pandas
- sklearn
- sqlalchemy
- pickle


## Project Structure <a name="structure"></a>

    data:
        - process_data.py (The script contians the ETL pipeline) 
        - disaster_categories.csv (Categories dataset)
        - disaster_messages.csv (Messages dataset)
      
    models:
        - train_classifier.py (The script contians the ML pipeline) 
      
    app:
        - templates
            - go.html (Shows the results of the model classification for the input message)
            - master.html (The web app home page) 
        - run.py


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


