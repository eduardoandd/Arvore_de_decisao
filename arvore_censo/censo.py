from sklearn import tree
import matplotlib.pyplot as plt
import shutil
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

arquivo='C:/Users/edubo/Desktop/I.A/pre-processamento/census.pkl'
destino='C:/Users/edubo/Desktop/I.A/Arvore_de_decisao/arvore_censo'
shutil.copy(arquivo,destino)

with open('census.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste=pickle.load(f)

arvore_censo= DecisionTreeClassifier(criterion='entropy')
arvore_censo.fit(X_treinamento,y_treinamento)
arvore_censo.feature_importances_
previsao=arvore_censo.predict(X_teste)
accuracy_score(y_teste,previsao)

cm=ConfusionMatrix(arvore_censo)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)

# fig,eixos=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
# tree.plot_tree(arvore_censo,filled=True)