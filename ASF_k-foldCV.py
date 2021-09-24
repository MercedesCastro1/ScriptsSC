import git
import os
import os.path as op

# Repo path
repo = git.Repo('.', search_parent_directories=True)

# import packages
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/paquetes.py'))).read())
# source specify parameters
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/ASF_specifyParameters.py'))).read())


# Set number of folds
K = 5

# Load data frame
name = 'dfToModel20200421_1134.csv'
path_modeling_sets = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/', name))
dfToModel = pd.read_csv(path_modeling_sets, header=0, sep=';', decimal=",", engine='python')

# Timestamp
timestr = time.strftime("%Y%m%d-%H%M")
# File names
trainName_prefix = 'dfASFTrain_'
sufix = '_'+timestr+'.csv'
testName_prefix = 'dfASFTest_'

# Separate target
x_dfToModel = dfToModel.drop(columns='default')
y_dfToModel = dfToModel[['default']]

# K-fold
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=1974)
skf.get_n_splits(x_dfToModel, y_dfToModel)
print(skf)

# Clasificador arbol
arbol_completo = tree.DecisionTreeClassifier()
arbol_reducido = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=30)
i=1
scores_full=[] #para guardar los resultados de los modelos
scores_short=[]

for train_index, test_index in skf.split(x_dfToModel, y_dfToModel):
    # split
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_dfToModel.iloc[train_index, :], x_dfToModel.iloc[test_index, :]
    y_train, y_test = y_dfToModel.iloc[train_index, :], y_dfToModel.iloc[test_index, :]
    # guardado de conjuntos
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    trainName = trainName_prefix + str(i)+'_of_'+str(K)+sufix
    trainPath = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/01. train sets', trainName))
    testName = testName_prefix + str(i) + '_of_' + str(K) + sufix
    testPath = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', testName))
    train.to_csv(trainPath, sep=";", decimal=",")
    test.to_csv(testPath, sep=";", decimal=",")
    # fit
    a = arbol_completo.fit(X_train[predictores], y_train)
    b = arbol_reducido.fit(X_train[predictores], y_train)
    # plots
    tree.plot_tree(a)
    dot_data = tree.export_graphviz(a, out_file=None,
                                    feature_names=predictores)
                                    #class_names=iris.target_names)
    graph = graphviz.Source(dot_data)
    graph.render(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/graficos', 'full_numeric_tree'+str(i))))
    dot_data = tree.export_graphviz(b, out_file=None,
                                    feature_names=predictores)
                                    #class_names=iris.target_names)
    tree.plot_tree(b)
    graph = graphviz.Source(dot_data)
    graph.render(
        op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/graficos', 'short_numeric_tree' + str(i))))

    #guardado del arbol
    treefileName_a = str(i)+"_ASFdecision_tree_full_"+timestr+".pkl"
    treefileName_b = str(i) + "_ASFdecision_tree_reduc_" + timestr + ".pkl"
    pickle.dump(a, open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', treefileName_a)), 'wb'))
    pickle.dump(b,
                open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', treefileName_b)),
                     'wb'))
    i=i+1
    scores_full.append(arbol_completo.score(X_test[predictores], y_test))
    scores_short.append(arbol_reducido.score(X_test[predictores], y_test))

print(scores_full)
print(scores_short)




