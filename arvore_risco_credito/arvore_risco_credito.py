from sklearn.tree import DecisionTreeClassifier #classificador baseado em árvore de decisão.
import shutil
import pickle
from sklearn import tree
import matplotlib.pyplot as plt

# arquivo='C:/Users/edubo/Desktop/I.A/naive_bayes/risco_credito/risco_credito.pkl'
# destino='C:/Users/edubo/Desktop/I.A/Arvore_de_decisao/arvore_risco_credito'
# shutil.copy(arquivo,destino)

with open('risco_credito.pkl', 'rb') as f:
    X,y=pickle.load(f)


arvore_risco_credito=DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(X,y) #treinamento
arvore_risco_credito.feature_importances_ # historia, divida, garantia, renda
previsores=['história de crédito','dívida','garantias','renda']
figure,eixos=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
tree.plot_tree(arvore_risco_credito,feature_names=previsores,class_names=arvore_risco_credito.classes_, filled=True)

previsoes=arvore_risco_credito.predict([[1,1,1,1]]) # historia, divida, garantia, renda