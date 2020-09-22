from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Graphviz2.38\bin'

data_set = pd.read_csv('customer_wait.csv', header = None)

X = data_set.iloc[:,:-1]
Y = data_set.iloc[1:,-1]
#print(X)

header_names = data_set.iloc[0,:]
#header_names = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est']
#print(feature_names)

target_names = data_set.iloc[1:,-1]
#target_names = ['Yes', 'No']

#print(class_names)


####VECTORISED DATA####

#Dictionary for column data
X_dict = X.T.to_dict().values()

#Dictionary to numpy array
vector_data = DictVectorizer(sparse=False)
X_vector = vector_data.fit_transform(X_dict)

#print(vector_data.get_feature_names())

X_Train = X_vector[:-1]
X_Test = X_vector[-1:] 

#print(X_Test)

label_encoder = LabelEncoder()
Y_Train = label_encoder.fit_transform(Y)

#print(Y_Train)

dcs_tree_no_fit = DecisionTreeClassifier(criterion='entropy')
dcs_tree_fit = dcs_tree_no_fit.fit(X_Train, Y_Train)

#Plotting the Decision tree
'''plt.figure(figsize=(15,10))
a = plot_tree(dcs_tree, 
              feature_names=header_names, 
              class_names=target_names, 
              filled=True, 
              rounded=True, 
              fontsize=14)

plt.show()'''

from sklearn.tree import export_graphviz
import graphviz

import pydot
import pyparsing


plot_desc_tree = graphviz.Source(export_graphviz(dcs_tree_fit))
plot_desc_tree.view()
#plot_tree.render('dtree_render_'+clf_name,view=True)

from IPython.core.display import display 

#Image(filename='decision_tree.png')

'''
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

dot_data = StringIO() 
plot_tree.export_graphviz(dcs_tree_fit, out_file=dot_data) 
plot_tree = pydot.graph_from_dot_data(dot_data.getvalue()) 
plot_tree.write_png('decision_tree.png') 


from IPython.core.display import Image 
Image(filename='data/vertebrate/tree.png')
'''
