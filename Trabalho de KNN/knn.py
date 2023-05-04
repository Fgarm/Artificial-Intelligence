import pandas as pd
from sys import argv
import numpy as np


class knn:
    def __init__(self, dados) -> None:
        self.treino = pd.read_csv(dados, sep=' ', header=None)
        self.classes = self.treino[self.treino.columns[-1]].to_numpy()
        self.nlinhas = len(self.treino)
        self.treino = self.treino.iloc[:,:-1].to_numpy()
        self.acertos = 0
        self.predicoes = 0
        
        uniqclasses = np.unique(self.classes)
        self.matriz_confusao = pd.DataFrame(0, columns=uniqclasses, index=uniqclasses)


    def predict_dataset(self, k: int, dados: pd.DataFrame, tipo_dist: str):
        classes_previsto = []
        for row in dados.itertuples(index=False):
            class_true = row[-1]
            distancias = np.tile(row[:-1], (self.nlinhas,1))
            distancias = distancias - self.treino
            if tipo_dist == 1:
                distancias = distancias[:,:]**2
            distancias = np.add.reduce(distancias, axis=1)
            if tipo_dist == 1:
               distancias = np.sqrt(distancias)
            k_indices = np.argsort(distancias)[:k]
            previsoes = [self.classes[i] for i in k_indices]
            previsao = max(set(previsoes), key=previsoes.count)
            classes_previsto.append(previsao)
            self.predicoes += 1
            if previsao == class_true:
                self.acertos += 1
            self.matriz_confusao[previsao][class_true] += 1
            
        return classes_previsto
    def taxa_de_acerto(self):
        return self.acertos / self.predicoes
    def resetar_modelo(self):
        self.acertos = 0
        self.predicoes = 0
        self.matriz_confusao =  pd.DataFrame(0, columns=self.matriz_confusao.columns, index=self.matriz_confusao.columns)

if __name__ == "__main__":
    try:
        if(argv[2].lower() == "euclediana" or argv[2].lower() == "euclidiana"):
            tipo_dist = 1
        else:
            print(argv[2].lower())
            tipo_dist = 2
        treino = argv[3]
        teste= argv[4]
    except IndexError:
        # Ex: python .\knn.py 1 euclidiana treinamento.txt teste.txt
        print("Modo de uso:\npython knn.py (k) (tipo de distancia) (caminho dados treino) (caminho dados teste)")
        exit()
    k = int(argv[1])
    teste = pd.read_csv(teste, sep=" ", header=None)
    model = knn(treino)

    #for k in [1,3,5,7,9,11,13,15,17,19]: 
    
    model.predict_dataset(k, teste, tipo_dist)
    print("k:",k," taxa de acerto:",model.taxa_de_acerto())
    print(model.matriz_confusao)
    model.resetar_modelo()