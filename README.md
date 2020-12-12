# Disaster Classifier

This project is a web app where an emergency worker can input a new message and get classification results in several categories. The proceess of classification is made by taking real messages that were sent during disaster events for training a machine learning pipeline to categorize the events. This will help to re send the messages to an appropriate disaster relief agency.

The web app will also display visualizations of the training data related to the genre and the categories of the messages.

# Project structure
<ul>
<li>notebooks: We've provided Jupyter notebooks in Project Workspaces with instructions to get you started with both data pipelines. The Jupyter notebook is not required for submission, but highly recommended to complete before getting started on the Python script.</li>
<li>workspace: After you complete the notebooks for the ETL and machine learning pipeline, you'll need to transfer your work into Python scripts, process_data.py and train_classifier.py. If someone in the future comes with a revised or new dataset of messages, they should be able to easily create a new model just by running your code. These Python scripts should be able to run with additional arguments specifying the files used for the data and model..</li>
</ul>

# Dataset and libraries

## Dataset

disaster_messages.csv: This data set has the message identification, another column with the message in its original language and finally a column with the message translated on English.
disaster_categories.csv: This file has all the message with the category classification.

## Libraries

<ul>
<li>pandas.</li>
<li>nltk.</li>
<li>sqlalchemy.</li>  
<li>re.</li> 
<li>sklearn.</li>
<li>pickle.</li> 
</ul>


# Results

This analysis shows the importance of the entire process and the CRISP-DP methodology, it also denotes the relevant factors when it comes to creating a business case. Additionally, it allows determining the relevant factors that determine the price of the accommodations posted on Airbnb, drawing up a preventive model on this.
