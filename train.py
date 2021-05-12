# importing all the important libraries
import re
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer

sw = stopwords.words('english')
# read the positvie text data
pos_rev = pd.read_csv('netflix\pos.txt' , sep='\n' , header = None , encoding = 'latin-1')

# adding a tareget column
pos_rev['mood'] = 1.0
pos_rev = pos_rev.rename(columns = {0:'review'})

# read the negative text data
neg_rev = pd.read_csv(r'C:\DataScience\NLP\Netflix sentiment analysis\netflix\negative.txt' , sep='\n' , header = None , encoding = 'latin-1')

# adding a tareget column
neg_rev['mood'] = 0.0
neg_rev = neg_rev.rename(columns = {0:'review'})

# cleaning the data

# 1. lower
# 2. remove spaces
# 3. punctuation
# 4. stopwords
# 5. lemmatize

pos_rev.loc[: , 'review'] = pos_rev.loc[: , 'review'].apply(lambda x : x.lower())
pos_rev.loc[: , 'review'] = pos_rev.loc[: , 'review'].apply(lambda x : re.sub(r"@\S+" , "" , x))
pos_rev.loc[: , 'review'] = pos_rev.loc[: , 'review'].apply\
(lambda x : x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
pos_rev.loc[: , 'review'] = pos_rev.loc[: , 'review'].apply\
(lambda x : " ".join([word for word in x.split() if word not in (sw)]))

neg_rev.loc[: , 'review'] = neg_rev.loc[: , 'review'].apply(lambda x : x.lower())
neg_rev.loc[: , 'review'] = neg_rev.loc[: , 'review'].apply(lambda x : re.sub(r"@\S+" , "" , x))
neg_rev.loc[: , 'review'] = neg_rev.loc[: , 'review'].apply\
(lambda x : x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
neg_rev.loc[: , 'review'] = neg_rev.loc[: , 'review'].apply\
(lambda x : " ".join([word for word in x.split() if word not in (sw)]))

# concatunatiing the pos and negative data
com_rev = pd.concat([pos_rev , neg_rev] , axis = 0).reset_index()

# train_test_split
X_train , X_test, y_train, y_test = train_test_split\
(com_rev['review'].values , com_rev['mood'].values , test_size = 0.2 , random_state = 101)

train_data = pd.DataFrame({'review':X_train , 'mood':y_train})
test_data = pd.DataFrame({'review':X_test , 'mood':y_test})

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data['review'])
test_vectors = vectorizer.transform(test_data['review'])

from sklearn import svm
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear')
classifier.fit(train_vectors , train_data['mood'])

pred = classifier.predict(test_vectors)

report = classification_report(test_data['mood'] , pred , output_dict=True)
# print(f"positive {report['1.0']['recall']}")
# print(f"negative {report['0.0']['recall']}")

# methods to save the model

# 1. pickle
# 2. joblib

# save the model using joblib
import joblib
model_file_name = 'netflix_svm_model.pkl'
vectorizer_filename = 'netflix_vector.pkl'
joblib.dump(classifier , model_file_name)
joblib.dump(vectorizer , vectorizer_filename)

# loading a model
vect = joblib.load('netflix_vector.pkl')
clf = joblib.load('netflix_svm_model.pkl')

# a = 'best movie ever seen'
# b = vect.transform([a])
# pred=clf.predict(b)
# if round(pred[0],2)==1.0:
#     output = 'good'
# else:
#     output = 'bad' 
# print(clf.predict(b))
# print(output)