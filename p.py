# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import *


# with open("C:\\Users\\Biplob\\Desktop\\bd.txt", "r", encoding="utf8") as file:
#     data = file.read()
#     data = data.split('\n')

# X = []
# y = []
# for line in data:
#     d = (line.split("\t"))
#     X.append(d)


# X_train = []
# y_train = []
# for i in range(len(X)):
#     train = X[i][0]
#     train_y = X[i][1]
# #     print(train_y)
#     X_train.append(train)
#     y_train.append(train_y)





# while True:
#     Pipe = Pipeline([
#     ('bow',TfidfVectorizer()),
#     ('tfidf',TfidfTransformer()),
#     ('classifier', LogisticRegression())
#     ])

#     Pipe.fit(X_train, y_train)
#     inp = input("Text: ")
#     print(Pipe.predict([f"{inp}"])[0])

import os
print(os.path.join("static/capture_image/", "".join(os.listdir("static/capture_image/")[-1:])))