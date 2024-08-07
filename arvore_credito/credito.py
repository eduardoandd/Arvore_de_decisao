import shutil
import pickle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix

# arquivo= 'C:/Users/edubo/Desktop/I.A/pre-processamento/credit.pkl'
# destino='C:/Users/edubo/Desktop/I.A/Arvore_de_decisao/arvore_credito'
# shutil.copy(arquivo,destino)

with open('credit.pkl', 'rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste=pickle.load(f)

arvore_credit= DecisionTreeClassifier(criterion='entropy',random_state=0)
arvore_credit.fit(X_treinamento,y_treinamento)
arvore_credit.feature_importances_ # income, age, loan
previsoes=arvore_credit.predict(X_teste)
y_teste
accuracy_score(y_teste,previsoes)

cm=ConfusionMatrix(arvore_credit)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)

previsores=['income','age','loan']
figure,eixos=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
tree.plot_tree(arvore_credit,feature_names=previsores,class_names=['0','1'],filled=True)
previsao=arvore_credit.predict([[0.10419945,0.54265932,1.17662679]])