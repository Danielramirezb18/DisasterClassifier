# Disaster Classifier

This project is a web app where an emergency worker can input a new message and get classification results in several categories. The proceess of classification is made by taking real messages that were sent during disaster events for training a machine learning pipeline to categorize the events. This will help to re send the messages to an appropriate disaster relief agency.

The web app will also display visualizations of the training data related to the genre and the categories of the messages.

# Project structure
<ul>
<li>notebooks: This notebook are an exploratory analysis of the data processing and mdoeling process</li>
<li>workspace: It contains all the files nedeed to execute the web app, aditionaly it has files of data transaformation and model building.</li>
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
