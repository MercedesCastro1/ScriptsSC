
# AJUSTE DEL MODELO: arbol de decision
# version construida: 15/04/2020
import git
import os
import os.path as op
import time
import pickle
import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Repo path
repo = git.Repo('.', search_parent_directories=True)

# import packages
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/paquetes.py'))).read())
# source specify parameters
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/ASF_specifyParameters.py'))).read())

# Timestamp
timestr = time.strftime("%Y%m%d-%H%M")

# Set file names
trainset_fileName = 'OHEdfTrain20200422_2207.csv'
testset_fileName = 'OHEdfTest20200422_2207.csv'

# Import test and train sets
path_train_sets = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/01. train sets', trainset_fileName))
path_test_sets = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', testset_fileName))
train = pd.read_csv(path_train_sets, header=0, sep=';', decimal=",", engine='python')
test = pd.read_csv(path_test_sets, header=0, sep=';', decimal=",", engine='python')

# Separate the target
x_train = train[predictores]
y_train = train[['default']]

x_test = test[predictores]
y_test = test[['default']]

# Fit Model
arbol = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=30)
a = arbol.fit(x_train[predictores], y_train)

# Graficar arbol
# plots
tree.plot_tree(a)
dot_data = tree.export_graphviz(a, out_file=None,
                                feature_names=predictores,
                                class_names=['not_default', 'default'])
graph = graphviz.Source(dot_data)
graph.render(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/graficos', 'tree_categ_'+timestr)))

# Save model
treefile = "ASF_decision_tree_categ"+timestr+".pkl"
pickle.dump(a, open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', treefile)), 'wb'))



