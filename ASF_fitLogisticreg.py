
## Ajuste regresión logísitica primer versión.
# Carga datos para modelar, separa en train y test y ajusta modelo

# Arranque
import os.path as op
import git

####### INTENTO DE EXTENSION DE CLASE PARA CALCULAR T TEST Y P VALUE
from sklearn import linear_model
from scipy import stats
import numpy as np

class LogisticRegression(linear_model.LogisticRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def fit(self, X, y, n_jobs=1):
        self = super(LogisticRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self
#######


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
logging.info('Entrenamiento de ASF utilizando script ASF_fitLogisticreg.py')
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
dfTrainCategorical
dictCategoricalCols
X_train_OHE = pd.concat([dfTrainCategorical.reset_index(drop=True), dfTrainNumeric.reset_index(drop=True)], axis=1)
#X_train_OHE = X_train_OHE.apply(lambda x: x.astype(float))

dfTestCategorical = getDfEncoded(encoder, X_test[predictores_categoricos], dictCategoricalCols )
dfTestNumeric = X_test[predictores_numericos]
X_test_OHE = pd.concat([dfTestCategorical.reset_index(drop=True), dfTestNumeric.reset_index(drop=True)], axis=1)
#X_test_OHE = X_test_OHE.apply(lambda x: x.astype(float))

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

# Prueba
import statsmodels.api as sm
X_train_ = sm.add_constant(X_train_OHE)
logit_model=sm.Logit(y_train.reset_index(drop=True), X_train_.reset_index(drop=True))
result=logit_model.fit()
print(result.summary2())

# Fit
clf_logistic = LogisticRegression()
#ravel transforma dataframe en un arreglo de 1D
clf_logistic.fit(X_train_OHE, np.ravel(y_train))

# Print the parameters of the model
print(clf_logistic.get_params())
# Print the intercept of the model
print(clf_logistic.intercept_)
# Print the models coefficients
print(clf_logistic.coef_)
print(X_train_OHE.columns)

# Pretty Printing Coef
coefs = {'predictor': X_train_OHE.columns, 'coef': clf_logistic.coef_.flatten()}
coeficientes = pd.DataFrame(data=coefs)
print(coeficientes)


#p = pd.DataFrame({'predictor': predictores})
#c = pd.DataFrame({'coef': clf_logistic.coef_}, index=[0])
#coef = pd.concat(p, clf_logistic.coef_)
logging.info('Parámetros del modelo: %s', clf_logistic.get_params())
logging.info('Intercepto: %s ', clf_logistic.intercept_)
logging.info('Coeficientes: \n %s ', coeficientes)
logging.info('Predictores Encoded: %s ', X_train_OHE.columns)

# Predict
prob_pred = clf_logistic.predict_proba(X_test_OHE)

# Guardar modelo
regressionfile = "ASF_logisticReg_"+timestr+".pkl"
pickle.dump(clf_logistic,
            open(op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', regressionfile)), 'wb'))
logging.info('Modelo de regresión entrenado: %s', regressionfile)

# Algunos kpis rapidos
acc=clf_logistic.score(X_test_OHE, y_test)
print(acc)
logging.info('Accuracy: %s', acc)
# ROC
fallout, sensitivity, thresholds = roc_curve(y_test, prob_pred[:, 1])
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_pred[:, 1])
print(auc)
logging.info('AUC: %s', auc)
