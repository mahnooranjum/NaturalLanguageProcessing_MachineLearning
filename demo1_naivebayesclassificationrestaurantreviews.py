##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    References:
#        SuperDataScience,
#        Official Documentation
##############################################################################
# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(y_test,y_pred):
  # Making the Confusion Matrix
  from sklearn.metrics import confusion_matrix
  con_mat = confusion_matrix(y_test, y_pred)
  print("===================================================")
  print(con_mat)
  from sklearn.metrics import classification_report
  print("===================================================")
  print(classification_report(y_test, y_pred))
  print("===================================================")

  # Get accuracy
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  print('Accuracy score: ', format(accuracy_score(y_test, y_pred)))
  print('Precision score: ', format(precision_score(y_test, y_pred)))
  print('Recall score: ', format(recall_score(y_test, y_pred)))
  print('F1 score: ', format(f1_score(y_test, y_pred)))
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

evaluate(y_test,y_pred)