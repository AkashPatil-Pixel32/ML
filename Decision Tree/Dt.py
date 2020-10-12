# Import modules
import pandas as pd 
import numpy as np
import pydotplus
import matplotlib.pyplot as plt 
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO

# Dummy Dataset
dataset =StringIO('''Plays Fetch,Is grumpy,Favorite food,Species
Yes,No,Bacon,Dog
No,Yes,Dog Food,Dog
No,Yes,Cat food,Cat
No,Yes,Bacon,Cat
No,No,Cat food,Cat
No,Yes,Bacon,Cat
No,Yes,Cat Food,Cat
No,No,Dog Food,Dog
No,Yes,Cat food,Cat
Yes,No,Dog Food,Dog
Yes,No,Bacon,Dog
No,No,Cat food,Cat
Yes,Yes,Cat food,Cat
Yes,Yes,Bacon,Dog
''')

#Load dataset
df = pd.read_csv(dataset)

df['B Plays Fetch']=np.where(df['Plays Fetch']=='Yes',True,False)
df['B Is grumpy']=np.where(df['Is grumpy']=='Yes',True,False)
df['B Favorite food']=[(0 if (food=='Bacon') else 1 if (food=='Dog Food') else 2) for food in df['Favorite food']]

