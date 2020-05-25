
# Dataset by : https://www.kaggle.com/venky73/spam-mails-dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

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

"""## Get the dataset"""

import pandas as pd
data = pd.read_csv('spam_ham_dataset.csv')
data.head()

y = data.label_num.values

X = data.text

sns.countplot(data = data, x = 'label');

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# STEP 1 :  REMOVE STOP WORDS
stop_words = set(stopwords.words('english')) 
X = X.apply(lambda email: ' '.join([ word for word in word_tokenize(email)  if not word in stop_words]))

# STEP 2 :  REMOVE SUBJECT
X = X.apply(lambda email: ' '.join([ word for word in word_tokenize(email)  if not word in ["Subject"]]))

# STEP 2 :  REMOVE NUMBERS
X = X.apply(lambda email: ' '.join([ word for word in word_tokenize(email)  if not word.isdigit()]))

# punctuations = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
# punctuations

# STEP 3 : REMOVE PUNCTUATIONS
# X = X.apply(lambda email: ' '.join([ word for word in word_tokenize(email) if not word in punctuations]))

X

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
X_bow = count_vector.fit_transform(X.astype(str))
#count_vector.get_feature_names()

X_bow  = X_bow.toarray()

X_bow

X_bow = pd.DataFrame(X_bow, columns = count_vector.get_feature_names())

X_bow['hey'].count()

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train.values
X_test = X_test.values

print(str(y_train.sum()) + " out of " + str(len(y_train)) + " were spam")



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()
evaluate(y_test,y_pred)

