
## Ajuste regresión logísitica: STATSMODELS
# Carga datos para modelar, separa en train y test y ajusta modelo

# Arranque
import os.path as op
import git

# Repo path
repo = git.Repo('.', search_parent_directories=True)

# import packages
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/paquetes.py'))).read())
# source specify parameters
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/ASF_specifyParameters.py'))).read())
# source encoder
exec(open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/EncoderFunctions.py'))).read())

# Timestamp
timestr = time.strftime("%Y%m%d-%H%M")

# Load data
dataPath = op.normpath(op.join(repo.working_tree_dir, 'data/02. clean', dataFileName))
dfToModel = pd.read_excel(dataPath, sheet_name='Sheet1')
# dfToModel = pd.read_csv(dataPath, header=0, sep=';', decimal=",", engine='python') # ver si es csv

# Escribir .log
logFile = 'ASF_logisticReg_'+ timestr+'.log'
logging.basicConfig(filename=op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', logFile)),
                    level=logging.INFO)
logging.info('Entrenamiento de ASF utilizando script ASF_fitLogisticregSTATSM.py')
logging.info('Base: %s ', dataFileName)

# Filtrar faltantes
dfToModel = dfToModel[dfToModel.faltante_respuesta == 0]

# Quitar 26 NAs (que quedaron dsp del filtro de faltantes)
dfToModel_no_NA = dfToModel[dfToModel.score_veraz.notnull()]

#Renombrar columnas
dfToModel_renamed = dfToModel_no_NA.rename(columns={'dictámen':'dictamen',
                                'funcional_veraz_categorías':'funcional_veraz_categorias'})

# Simple train test split
X = dfToModel_renamed.drop(respuesta, axis=1)
y = dfToModel_renamed[respuesta]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3,
                                                                            random_state=1974, stratify=y)
# Save train and test sets
train = pd.concat([X_train, y_train], axis=1, ignore_index=False)
test = pd.concat([X_test, y_test], axis=1, ignore_index=False)
nombre_train = 'train_set_'+timestr+'.csv'
nombre_test  = 'test_set_'+timestr+'.csv'
path_to_save_train = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/01. train sets', nombre_train))
path_to_save_test = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', nombre_test))
train.to_csv(path_to_save_train, sep=";", decimal=",")
test.to_csv(path_to_save_test, sep=";", decimal=",")

# Encoder Pablito
dfTrainCategorical, encoder = getDfAndEncoder(X_train[predictores_categoricos], dictCategoricalCols,flagUseStatsFormulaNameLike=False )
dfTrainNumeric = X_train[predictores_numericos]
X_train_OHE = pd.concat([dfTrainCategorical.reset_index(drop=True), dfTrainNumeric.reset_index(drop=True)], axis=1)

dfTestCategorical = getDfEncoded(encoder, X_test[predictores_categoricos], dictCategoricalCols )
dfTestNumeric = X_test[predictores_numericos]
X_test_OHE = pd.concat([dfTestCategorical.reset_index(drop=True), dfTestNumeric.reset_index(drop=True)], axis=1)

# Save train and test sets con OHE
train_OHE = pd.concat([X_train_OHE.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1, ignore_index=False)
test_OHE = pd.concat([X_test_OHE.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1, ignore_index=False)
nombre_train_OHE = 'train_set_OHE_'+timestr+'.csv'
nombre_test_OHE  = 'test_set_OHE_'+timestr+'.csv'
path_to_save_train_OHE = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/01. train sets', nombre_train_OHE))
path_to_save_test_OHE = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', nombre_test_OHE))
train_OHE.to_csv(path_to_save_train_OHE, sep=";", decimal=",")
test_OHE.to_csv(path_to_save_test_OHE, sep=";", decimal=",")

# Logging
logging.info('Split train: %s, test: %s', nombre_train, nombre_test)
logging.info('Versiones con OHE train: %s, test: %s', nombre_train_OHE, nombre_test_OHE)
logging.info('Variables utilizadas: %s ', predictores)
logging.info('Categorías de referencia: %s ', dictCategoricalCols)

# Fit
X_train_ = sm.add_constant(X_train_OHE)
logit_model = sm.Logit(y_train.reset_index(drop=True), X_train_.reset_index(drop=True))
result = logit_model.fit()
print(result.summary2())

logging.info(result.summary())

# Predict
X_test_OHE_ = sm.add_constant(X_test_OHE)
prob_pred = result.predict(X_test_OHE_)

# Confusion matrix
result.pred_table()

# Guardar modelo
regressionfile = "ASF_logisticReg_"+timestr+".pkl"
pickle.dump(result,
            open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', regressionfile)), 'wb'))
logging.info('Modelo de regresión entrenado: %s', regressionfile)

# Algunos kpis rapidos
# ROC
fallout, sensitivity, thresholds = roc_curve(y_test, prob_pred)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_pred)
print(auc)
logging.info('AUC: %s', auc)
