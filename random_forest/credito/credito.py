from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix

with open('credit.pkl', 'rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste=pickle.load(f)

random_forest_credit=RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=0)
random_forest_credit.fit(X_treinamento,y_treinamento)
previsoes=random_forest_credit.predict(X_teste)
accuracy_score(y_teste,previsoes)

cm=ConfusionMatrix(random_forest_credit)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)
