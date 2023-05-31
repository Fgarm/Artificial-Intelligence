from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn import svm
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches

# lendo dados do prof
treino = pd.read_csv("../treinamento.txt", sep=' ', header=None)
classes_treino = treino[treino.columns[-1]].to_numpy()
# print(classes_treino)
nlinhas = len(treino)
treino = treino.iloc[:,:-1].to_numpy()

teste = pd.read_csv("../teste.txt", sep=' ', header=None)
classes_teste = teste[teste.columns[-1]].to_numpy()
nlinhas_teste = len(teste)
teste = teste.iloc[:,:-1].to_numpy()

# lendo dados do sklearn
dados, classes = load_digits(return_X_y=True)


dados_prof = False
dados_prof = True
classes_possiveis = []
x_train, x_test, y_train, y_test = [],[],[],[]
if dados_prof:
    classes_possiveis = list(set(classes_teste))
    
    x, y = treino, classes_treino
    for c in classes_possiveis:
        idxs: np.ndarray = y == c
        x_train.append(x[idxs,:])
        y_train.extend(y[idxs])
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    x, y = teste, classes_teste

    for c in classes_possiveis:
        idxs: np.ndarray = y == c
        x_test.append(x[idxs,:])
        y_test.extend(y[idxs])
    x_test = np.concatenate(x_test)
    y_test = np.array(y_test)
else:
    classes_possiveis = list(set(classes))
    x, y = dados, classes
    xc = []
    yc = []
    for c in classes_possiveis:
        idxs: np.ndarray = y == c
        xc.append(x[idxs,:])
        yc.extend(y[idxs])
    xc = np.concatenate(xc)
    yc = np.array(yc)
    x_train, x_test, y_train, y_test = train_test_split(xc, yc, test_size=0.2)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

cluster_possiveis = [*range(1, 25)]
cluster_pedidos = [5, 10, 20]
pontuacoes_knn = []
pontuacoes_svm = []
pont_clustpedidos_knn = []
pont_clustpedidos_svm = []


fig, ax = pyplot.subplots() 


modelo = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
srv = svm.SVC(probability=True, kernel='linear')


modelo.fit(x_train, y_train)
score_knn = modelo.score(x_test, y_test)
l1 = ax.axhline(score_knn, c="blue", label="Baseline KNN", linestyle = '--',)
ax.text(0, score_knn-.03, score_knn, fontsize=9)


srv.fit(x_train, y_train)
score_svm = srv.score(x_test, y_test)
l2 = ax.axhline(score_svm, c="red", label="Baseline SVM", linestyle = '--',)
ax.text(0, score_svm+.005,score_svm, fontsize=9)
ax.legend(loc = 'lower left')


for n_clusters in cluster_possiveis:

    km = KMeans(n_clusters=n_clusters, n_init=20)
    centroides_classes = {}
    for digit_class in classes_possiveis:
        indexes = y_train == digit_class
        instancias = x_train[indexes]
        km.fit(instancias)
        centroides_classes[digit_class] = km.cluster_centers_
    centroides_para_treino = []
    centroides_labels = []
    for i in centroides_classes:
        for centroide in centroides_classes[i]:
            centroides_para_treino.append(centroide)
            centroides_labels.append(i)
        

    modelo.fit(centroides_para_treino, centroides_labels)
    score_knn = modelo.score(x_test, y_test)
    pontuacoes_knn.append(score_knn)
    if n_clusters in cluster_pedidos:
        pont_clustpedidos_knn.append(score_knn)
    #print(classification_report(y_test, modelo.predict(x_test)))
    print("\r#",end="", flush=True)
    print("--"*n_clusters, end="")
    print("  "*(cluster_possiveis[-1] - n_clusters), end="")
    print("#", end="" ,flush=True)
    srv.fit(centroides_para_treino, centroides_labels)
    score_svm = srv.score(x_test, y_test)
    pontuacoes_svm.append(score_svm)
    if n_clusters in cluster_pedidos:
        pont_clustpedidos_svm.append(score_svm)
    #print(classification_report(y_test, srv.predict(x_test)))

print("")

ax.plot(cluster_possiveis, pontuacoes_knn, c="blue")
ax.plot(cluster_possiveis, pontuacoes_svm, c="red")
for i in range(len(cluster_pedidos)):
    ax.scatter(cluster_pedidos[i], pont_clustpedidos_knn[i], c="blue")
    ax.text(cluster_pedidos[i]-.02, pont_clustpedidos_knn[i]-.02, pont_clustpedidos_knn[i], fontsize=9)
    ax.scatter(cluster_pedidos[i], pont_clustpedidos_svm[i], c="red")
    ax.text(cluster_pedidos[i]-.02, pont_clustpedidos_svm[i]-.02, pont_clustpedidos_svm[i], fontsize=9)
ax.set_xlabel('Número de clusters')
ax.set_ylabel('Pontuações obtidas')
ax.set_title('Precisão dos modelos por número de clusters')
ax.legend(handles=[l1, l2, mpatches.Patch(color='red', label='SVM'), mpatches.Patch(color='blue', label='KNN')])
ax.legend(loc = 'lower left')
pyplot.show()