
# REPORTE DE METRICAS DEL MODELO
# version construida: 15/04/2020
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

# Set file names
testset_fileName = 'test_set_20200629-1127.csv'
testset_OHE_fileName = 'test_set_OHE_20200629-1127.csv'
model_fileName = 'ASF_logisticReg_20200629-1127.pkl'

# Import test set
path_test_set = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', testset_fileName))
path_test_set_OHE = op.normpath(op.join(repo.working_tree_dir, 'data/03. modeling sets/02. test sets', testset_OHE_fileName))
test = pd.read_csv(path_test_set, header=0, sep=';', decimal=",", engine='python')
test_OHE = pd.read_csv(path_test_set_OHE, header=0, sep=';', decimal=",", engine='python')

# OHE
x_test = ''
x_test_OHE = test_OHE.drop(respuesta, axis=1)
x_test_OHE = x_test_OHE.drop(test_OHE.columns[0], axis=1)
y_test = test_OHE[respuesta]

# Import model
model_path = op.normpath(op.join(repo.working_tree_dir, 'code/04 Fit Models/01 Model objects', model_fileName))
model = pickle.load(open(model_path, 'rb'))

# Predict classes and prob
x_test_OHE_ = sm.add_constant(x_test_OHE)
y_pred_prob = model.predict(x_test_OHE_)
y_05pred = (model.predict(x_test_OHE_) >= 0.5).astype(int)

preds = {'default_pred': y_05pred, 'PD_pred': y_pred_prob}
predictions = pd.DataFrame(data=preds)

# Root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))
print("RMSE")
print(rmse)

# Mean squared error
mse = mean_squared_error(y_test, y_pred_prob)
print("MSE")
print(mse)

# Normalized mean squared error
mean = sum(y_test.default_periodo)/y_test.size
df_mean_pred = pd.DataFrame({'default': [mean]*y_test.size})
mse_media = mean_squared_error(y_test, df_mean_pred)
nmse = mse / mse_media
print("NMSE")
print(nmse)


# Confusion matrix
print("Punto de corte 0.5")
skplt.metrics.plot_confusion_matrix(y_test, y_05pred, labels=None)
plt.title('Punto de corte 0.5')
plt.show()

print("Punto de corte 0.1295")
y_pred_prueba = (model.predict(x_test_OHE_) >= 0.1295).astype(int)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_prueba, labels=None)
plt.title('Punto de corte 0.1295')
plt.show()

print("Punto de corte 0.155")
y_pred_prueba155 = (model.predict(x_test_OHE_) >= 0.155).astype(int)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_prueba155, labels=None)
plt.title('Punto de corte 0.155')
plt.show()

print("Punto de corte 0.135")
y_pred_prueba135 = (model.predict(x_test_OHE_) >= 0.135).astype(int)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_prueba135, labels=None)
plt.title('Punto de corte 0.135')
plt.show()

print("Punto de corte 0.1626")
y_pred_prueba1626 = (model.predict(x_test_OHE_) >= 0.1626).astype(int)
skplt.metrics.plot_confusion_matrix(y_test, y_pred_prueba1626, labels=None)
plt.title('Punto de corte 0.1626')
plt.show()


# PRODUCTIVO

# con estado recategorizado
test['estado_recategorizado_num'] = test['estado_recategorizado'].replace(['Aceptado'], 0)
test['estado_recategorizado_num'] = test['estado_recategorizado_num'].replace(['No aceptado'], 1)
skplt.metrics.plot_confusion_matrix(y_test, test.estado_recategorizado_num, labels=None)
plt.title('Modelo en productivo')
plt.xlabel('Estado_recategorizado (0 Aceptado - 1 No aceptado)')
plt.ylabel('default_periodo')
plt.show()

# con punto de corte en PD calculada
scoresName = 'ASF_Validacion_RL_20200630_1057.csv'
path_scores = op.normpath(op.join(repo.working_tree_dir, 'code/05 Performance', scoresName))
scores = pd.read_csv(path_scores, header=0, sep=';', decimal=",", engine='python')

scores['pred155'] = scores['PD_calculada'].apply(lambda x: 1 if x > 0.155 else 0)
skplt.metrics.plot_confusion_matrix(y_test, scores.pred155, labels=None)
plt.title('Modelo en productivo: punto de corte de PD = 0.155')

scores['pred135'] = scores['PD_calculada'].apply(lambda x: 1 if x > 0.135 else 0)
skplt.metrics.plot_confusion_matrix(y_test, scores.pred135, labels=None)
plt.title('Modelo en productivo: punto de corte de PD = 0.135')

scores['pred155_Killers'] = scores['PD_calculada_con_killers'].apply(lambda x: 1 if x > 0.155 else 0)
skplt.metrics.plot_confusion_matrix(y_test, scores.pred155_Killers, labels=None)
plt.title('Modelo en productivo: punto de corte de PD = 0.155')

scores['pred135_Killers'] = scores['PD_calculada_con_killers'].apply(lambda x: 1 if x > 0.135 else 0)
skplt.metrics.plot_confusion_matrix(y_test, scores.pred135_Killers, labels=None)
plt.title('Modelo en productivo: punto de corte de PD = 0.135')


# ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("AUC")
print(roc_auc)

plt.clf()
plt.plot(fpr, tpr)
lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


# Df con rdos
nombre_df = 'predicciones'+timestr+'.csv'
path_df = op.normpath(op.join(repo.working_tree_dir, 'code/05 Performance', nombre_df))
rdos = pd.concat([test, predictions], axis=1)
rdos.to_csv(path_df, sep=";", decimal=",")

# KS
# contra respuesta observada: "default"
print("two-sample K-S test")
print(ks_2samp(rdos.default_periodo, rdos.PD_pred))
# Gráfico para validar
rdos[['PD_pred']] = rdos.PD_pred.round(decimals = 3)
rdos['deciles'] = pd.qcut(rdos['PD_pred'], 10)
# cada decil tiene 41 registros
cant_default_periodos = rdos.groupby('deciles')['default_periodo', 'PD_pred'].sum()
cumsum = cant_default_periodos.cumsum()

# Grafico comparando prob deciles
rdos[['PD_pred']] = rdos.PD_pred.round(decimals = 3)
rdos['deciles'] = pd.qcut(rdos['PD_pred'], 10)
df_scatter = rdos.groupby('deciles', as_index = False).mean()
df_scatter['deciles'] = df_scatter.deciles.astype(str)
ax1 = df_scatter.plot.scatter(x='deciles',
                               y='default_periodo',
                               c='DarkBlue',
                               title='PD predicha VS proporción real de default_periodo',
                               fontsize=6,
                               rot=90)
ax1.set_xlabel("Deciles PD (predicha)")
ax1.set_ylabel("Proporción de default_periodo real por decil")


