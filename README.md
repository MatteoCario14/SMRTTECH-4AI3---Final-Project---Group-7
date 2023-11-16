# SMRTTECH-4AI3-Final-Project-Group-7
This repository contains the code for a final project based on 'Fake news Detection' for the course SMRTTECH 4AI3.
# Description:
Social media has provided an excellent interactive technology platform that
allows the creation, sharing, and exchange of interests, ideas, or information via
virtual networks very quickly. A new platform enables endless opportunities for
marketing to reach new or existing customers. However, it has also opened the
devil’s door for falsified information which has no proven source of
information, facts, or quotes. It is really hard to detect whether the given news
or information is correct or not. Here, as a part of this project, we need to detect
the authenticity of given news using DL.
The dataset contains around 7k fake news, including a title, body, and label
(FAKE or REAL). The task is to train the data to predict if the given news is
fake or real.
# Project Outcomes:
• Pre-process the data to remove stop words. Stop words are the most occurring words in the language. It’s necessary to filter that out first.

• Evaluate the various algorithms which can affect the best outcome

• Train a model to predict the likelihood of REAL news.

# Actual Outcomes:
Our final project for 4AI3 involves developing a fake news detection system. Our team, consisting of Matteo Cario, Haider Khan, Nur Tawfeeq, Amrit Maharaj, Daniel Graham, and Christina Mathews, is focusing on leveraging various machine learning models to differentiate between 'Fake' and 'Real' news. Here's an overview of what we are accomplishing:

**Packages:** We are utilizing Python libraries such as Pandas for data handling, NLTK for natural language tasks, along with Matplotlib and Seaborn for visualization, and Sklearn for its machine learning capabilities.

**Data Loading:** We load the dataset that was provided to us in SMRTTECH 4AI3, where we examine the distribution of labels to understand the balance between 'Fake' and 'Real' news instances.

**Preprocessing:** Our preprocessing step includes converting text to lowercase and stripping non-alphanumeric characters to standardize the data, followed by removing stopwords to filter out noise from the text data.

**Feature Extraction:** We apply TfidfVectorizer to turn the textual content into a numerical format that is amenable to machine learning algorithms.

**Model Training and Evaluation:** We split our dataset into a training set and a testing set, then train several models: Multinomial Naive Bayes, Logistic Regression, Random Forest, Support Vector Machine, and Gradient Boosting Classifier. We evaluate each model's performance using a classification report, confusion matrix, F1 score, and ROC-AUC score.

**Prediction:** We've also crafted a function that predicts whether a new news article is likely 'Fake' or 'Real' based on its title and text.

**Visualization:** To make our findings clear and accessible, we visualize the results of each model's performance, focusing on the confusion matrix and F1 score.

Our goal is to determine the most effective model for fake news detection, making use of natural language processing and machine learning techniques to analyze and classify news content.





