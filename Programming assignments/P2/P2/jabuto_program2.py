# Joshua Abuto 1001530342
# Data Mining Homework 2
# Disclaimer: My terminal was running the program earlier but when I started making some tweaks to my file it started
# giving me the same issues I had before I fixed it before.
# I don't know if my code works now but the syntax and everything looks right
import numpy as np
import pandas as pd


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# print the column names
original_headers = list(nba.columns.values)
print(original_headers)


# "Position (pos)" is the class attribute we are predicting.
class_column = 'Pos'

# The dataset contains attributes such as player name and team name.
# We know that they are not useful for classification and thus do not
# include them as features.
feature_columns = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'ORB', 'DRB','TRB', 'BLK']

# Pandas DataFrame allows you to select columns.
# We use column selection to split the data into features and class.
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

print(nba_feature[0:3])
print(list(nba_class[0:3]))

train_feature, test_feature, train_class, test_class = train_test_split(nba_feature, nba_class, stratify=nba_class, train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

# Naive Bayes Classifier
nb = GaussianNB().fit(train_feature, train_class)
prediction = nb.predict(test_feature)


train_class_df = pd.DataFrame(train_class, columns=[class_column])
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
train_data_df.to_csv('train_data.csv', index=False)

temp_df = pd.DataFrame(test_class, columns=[class_column])
temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
test_data_df.to_csv('test_data.csv', index=False)

print("NBC Results:")
print("Test set score: {:.3f}".format(nb.score(test_feature, test_class)))
print("Test set predictions:\n{}".format(prediction))
print("Test set accuracy: {:.2f}".format(nb.score(test_feature, test_class)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

scores = cross_val_score(nb, nba_feature, nba_class, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))




