# 4AI3 - Final Project - Group 7 - Fake News Detection
#  Haider Khan		    400272546
#  Matteo Cario		    400238834
#  Nur Tawfeeq		    400208770
#  Amrit Maharaj        400243522
#  Daniel Graham	    400272771
#  Christina Mathews    400256960

# Packages installed for this project
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from matplotlib.ticker import MultipleLocator


# Loading the dataset using the read_csv() from pandas
df = pd.read_csv('news.csv')  # This functions reads the file 'news.cvs' using the pandas library which is then stored in the variable 'df'
print(df['label'].value_counts())  # Printing the column 'label' from 'df' and using the function 'value_counts()' to obtain the distribution of 'Fake' and 'Real' entries

# Preprocessing the data using nltk: nltk will help us in text classification within the datasheet provided.
# The below code downloads a collection of stopwords from nltk such as 'and', 'is', 'the', etc., that will be filtered out before processing the data
nltk.download('stopwords')  # Source: https://www.nltk.org/api/nltk.downloader.html#module-nltk.downloader
                                    # https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
sw = set(stopwords.words('english'))  # This line retrieves the list of 'english' stopwords using nltk and stores them into the variable 'sw'


# The following defined function 'processing' will prepare the text data provided in 'news.csv'
def processing(text):
    # The text.lower() function on line 41 converts all characters to lowercase.
    # This helps the algorithm in standardizing the text to ensure that words such as 'The' and 'the' are treated as the same word
    text = text.lower() # Source: https://www.programiz.com/python-programming/methods/string/lower
    # The 'char.isalnum()' and 'char.isspace()' functions on line 44 help assist the algorithm in removing all
    # characters that are not alphanumeric (a-z, A-Z, 0-9) or spaces.
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Source: https://www.programiz.com/python-programming/methods/string/join
                                                                                 # Source: https://www.programiz.com/python-programming/methods/string/isspace
                                                                                 # Source: https://www.programiz.com/python-programming/methods/string/isalnum
                                                                                 # Source: https://stackabuse.com/removing-stop-words-from-strings-in-python/
    # The function 'text.split()' on line 50 splits the text into words, removes stopwords, and rejoins them back together into a string.
    # Stopwords like 'and', 'the', etc., are removed because they usually don't contain useful information for analysis.
    # The remaining words are then joined back into a single string, separated by spaces.
    text = ' '.join([word for word in text.split() if word not in sw])  # Source: https://www.programiz.com/python-programming/methods/string/split
    return text


df['c_text'] = df['title'] + ' ' + df['text']  # This creates a new column called 'c_text' which concatenates the text from both the 'title' and 'text' columns
df['c_text'] = df['c_text'].apply(processing)  # This line applies the defined function 'processing' to each row in 'c_text' using '.apply()' from pandas.

v = TfidfVectorizer()  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# This assigns the variable 'v' to the object 'TfidfVectorizer()', which was imported from sklearn.feature_extraction.text.
# The report will highlight more information about this specific 'Feature extraction' and why we choose it.

X_train, X_test, y_train, y_test = train_test_split(df['c_text'], df['label'], test_size=0.2, random_state=42)
# The above line uses 'train_test_split' to partition our dataset into training and testing subsets which are 'X_train', 'X_test', 'y_train', and 'y_test'.
# 'df['c_text']' and 'df['label']' are the parameters that are being passed into 'train_test_split'
# 'test_size=0.2' specifies that 20% of the dataset will be used for test while the remaining 80% will be used for training the model
# 'random_state=42' initializes the internal random number generator, which decides the splitting of the data into train and test indices.

# Selected models that will be used for this project
models = {
    # The model on line 71 uses text classification tasks involving discrete features such as word counts
    'Multinomial Naive Bayes': MultinomialNB(),  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    # The model on line 73 uses binary classification
    'Logistic Regression': LogisticRegression(),  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # The model on line 75 uses multiple learning algorithms to obtain a better predictive performance
    'Random Forest': RandomForestClassifier(),  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # The model on line 77 uses probability estimates, which are not available by default in SVC. The parameter 'probability=True' allows for ROC-AUC calculation.
    'Support Vector Machine': SVC(probability=True),  # Source: https://stackoverflow.com/questions/15015710/how-can-i-know-probability-of-class-predicted-by-predict-function-in-support-v
    # The model on line 79 uses a machine learning technique for regression and classification
    'Gradient Boosting': GradientBoostingClassifier()  # Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
}

# The following for loop will iterate through the 'models' that were selected. 'name' is the key and 'model' is the value
for name, model in models.items():
    pip = make_pipeline(v, model)
    # For each model, a pipeline 'pip' is created with 2 steps, feature transformation by calling the variable 'v' and the actual model that is used
    # This pipeline will automate the process of vectorizing the input data and then applying the model
    pip.fit(X_train, y_train)  # The pipeline is then trained using the '.fit()' function on the training data 'X_train', and 'y_train'
    y_pred = pip.predict(X_test)  # The pipeline will then be used to make prediction on the test data 'X_test' and then stored in 'y_pred'.

    # Evaluating the models
    print("Model: ", name)  # This prints the name of the model that is currently being evaluated
    print(classification_report(y_test, y_pred))  # This prints out a classification report that includes several metrics such as precision, recall and f1-score
    c_matrix = confusion_matrix(y_test, y_pred)  # This line calculates the confusion matrix for the model and assigns it the variable 'c_matrix'
    print("Confusion Matrix:\n", c_matrix)  # This prints the confusion matrix which shows the number of correct and incorrect predictions broken down by each class
    f1 = f1_score(y_test, y_pred, average='weighted')  # This line computes the weighted average of the F1-score for the selected model
    print("F1 Score: ", f1)  # This prints the calculated f1-score
    roc_auc = roc_auc_score(y_test, pip.predict_proba(X_test)[:, 1])  # Calculates the ROC AUC score. High scores represent better discrimination by the model between positive and negative classes
    print("ROC-AUC Score: ", roc_auc)  # This prints the calculated ROC-AUC score
    print("\n")  # This line allows for spacing between the models so that the console is more legible
    # Note: the ROC-AUC Score should only be for binary classification but in this case it displays for all models


# This function will be used to predict weather a given news article is classified as 'Fake' or 'Real'
def predict_news(news_title, news_text):
    c_text = processing(news_title + ' ' + news_text)  # This line concatenates 'news_title' and 'news_text' with a space and then applies the 'processing' function
    prediction = pip.predict([c_text])  # This line sends the pre-processed data into a list 'predict' which is stored in the variable 'prediction'
    return 'FAKE' if prediction[0] == 'FAKE' \
        else 'REAL'  # This is a ternary conditional statement that simplifies the if-else block into a single line


# Example prediction. We can test our model by inputting any example title and text from the datasheet
example_title = "Kerry to go to Paris in gesture of sympathy"
example_text = "U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week..."
print(predict_news(example_title, example_text))  # This will print 'Fake' or 'Real' depending on the inputted text using the defined function 'predict_news'


sns.set_style("whitegrid")  # This line sets the aesthetic style of the plots

# The following for loop iterates through 'models', creates a pipeline for each, trains them, makes predictions, and computes a confusion matrix
for name, model in models.items():
    pip = make_pipeline(v, model)  # This creates a pipeline named 'pip' for each model, including the 'v' vectorized model
    pip.fit(X_train, y_train)  # This line trains the pipeline on the data
    y_pred = pip.predict(X_test)  # This line uses the trained pipeline to make predictions on the test data
    c_matrix = confusion_matrix(y_test, y_pred)  # Generates the confusion matrix from the true labels 'y_test' and the predicited labels 'y_pred'

    plt.figure(figsize=(10, 7))  # Sets the size to 10 by 7 inches
    sns.heatmap(c_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])  # Creates the heat map for the confusion matrix
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

# The following code will calculate and store the F1-score of the different machine learning models that are selected.
f1_scores = []
model_names = []

for name, model in models.items():
    pip = make_pipeline(v, model)   # This creates a pipeline named 'pip' for each model, including the 'v' vectorized model
    pip.fit(X_train, y_train)   # This line trains the pipeline on the data
    y_pred = pip.predict(X_test)  # This line uses the trained pipeline to make predictions on the test data
    f1 = f1_score(y_test, y_pred, average='weighted')  # This line calculates the weighted F1 score comparing the true labels 'y_test' and the predicited labels 'y_pred'
    f1_scores.append(f1)  # This line and line 143 Appends the calculated F1 score and the model's name to their respective lists for the upcoming visulatization
    model_names.append(name)

plt.figure(figsize=(10, 7))
sns.barplot(x=f1_scores, y=model_names)
plt.xlabel('Weighted Average F1 Score')
plt.title('Comparison of Model F1 Scores')
# The following lines Set x-axis major tick frequency to 0.1 instead of the default 0.2
ax = plt.gca()  # Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.gca.html
ax.xaxis.set_major_locator(MultipleLocator(0.1))  # Source: https://matplotlib.org/stable/api/ticker_api.html
# The following line adjusts the plot so that the y-axis 'model' label is fully displayed
plt.tight_layout()  # Source: https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.tight_layout.html
plt.show()