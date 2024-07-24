from sklearn.tree import DecisionTreeClassifier #classificador baseado em árvore de decisão.
import shutil
import pickle
from sklearn import tree
import matplotlib.pyplot as plt

arquivo='C:/Users/edubo/Desktop/I.A/naive_bayes/risco_credito/risco_credito.pkl'
destino='C:/Users/edubo/Desktop/I.A/Arvore_de_decisao/arvore_risco_credito'
shutil.copy(arquivo,destino)

with open('risco_credito.pkl', 'rb') as f:
    X,y=pickle.load(f)
