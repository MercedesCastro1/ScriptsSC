
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import git
import os
import os.path as op

# Repo path
repo = git.Repo('.', search_parent_directories=True)

# import packages
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/paquetes.py'))).read())
# source specify parameters
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/ASF_specifyParameters.py'))).read())

# Timestamp
timestr = time.strftime("%Y%m%d-%H%M")

# Load data frame
name = 'dfToModel20200421_1134.csv'
path_modeling_sets = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/', name))
dfToModel = pd.read_csv(path_modeling_sets, header=0, sep=';', decimal=",", engine='python')

# Set number of folds
K = 10

# Separate target
x_dfToModel = dfToModel.drop(columns='default')
y_dfToModel = dfToModel[['default']]

# Scorer
def root_mean_squared_error(y_true, y_pred):
    """ Root mean squared error regression loss
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
pipe_tree = make_pipeline(tree.DecisionTreeClassifier())

# make an array of depths and num_leafs to choose from
depths = np.arange(3, 4, 5)
num_leafs = [20, 30, 40, 50]

param_grid = [{'decisiontreeclassifier__max_depth': depths, 'decisiontreeclassifier__min_samples_leaf': num_leafs}]
gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, scoring=rmse_scorer, cv=10)
gs.fit(x_dfToModel[predictores], y_dfToModel)

print(gs.best_score_)
print(-gs.best_score_)
print(gs.best_params_)

# # Set file names
# trainset_fileName = 'dfTrain20200422_2207.csv'
# testset_fileName = 'dfTest20200422_2207.csv'
# # import train and test to predict
# path_train_sets = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/01. train sets', trainset_fileName))
# path_test_sets = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', testset_fileName))
# train = pd.read_csv(path_train_sets, header=0, sep=';', decimal=",", engine='python')
# test = pd.read_csv(path_test_sets, header=0, sep=';', decimal=",", engine='python')
#
# # Separate the target
# x_train = train[predictores]
# y_train = train[['default']]
# x_test = test[predictores]
# y_test = test[['default']]
#
# # Fit Model
# best_model = gs.best_estimator_
# model = best_model.fit(x_train, y_train)
#
# # Predict
# y_predicted = model.predict(x_test)
# root_mean_squared_error(y_test, y_predicted)
#
# # Save model
# treefile = "ASF_optimized_decision_tree_"+timestr+".pkl"
# pickle.dump(model, open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', treefile)), 'wb'))
#
# # Graficar arbol
# # plots
# tree.plot_tree(model.named_steps['decisiontreeclassifier'])
# dot_data = tree.export_graphviz(model, out_file=None,
#                                 feature_names=predictores)
#                                 #class_names=['not_default', 'default'])
# graph = graphviz.Source(dot_data)
# graph.render(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/graficos', 'tree'+timestr)))