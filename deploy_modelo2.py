#importacao de dependencias 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve, classification_report,\
                            accuracy_score, confusion_matrix, auc

sales = pd.read_csv("dataset2.csv")

sales["Product_Category"] = sales["Product_Category"].astype("category")

# removendo da coluna Product category as categorias Accessories e Clothing
sales["Product_Category"] = sales["Product_Category"].cat.remove_categories("Accessories")

sales["Product_Category"] = sales["Product_Category"].cat.remove_categories("Clothing")

#limpando os NA's
salesbike=sales.dropna(subset=['Product_Category'])

salesbike = salesbike.drop(['Day','State','Unit_Cost', 'Unit_Price'], axis=1)

salesbike['Sub_Category'] = salesbike['Sub_Category'].astype('category') # transformando Sub_Category para categórico

#Definindo grupos por idade
salesbike['Customer_Gender'] = salesbike['Customer_Gender'] .map({'M': 1, 'F': 0})
salesbike['Age_Group'] = salesbike['Age_Group'] .map({'Youth (<25)': 1, 'Young Adults (25-34)': 0, 'Adults (35-64)':0, 'Seniors (64+)':0})

model = LogisticRegression(penalty='none', solver='lbfgs') #newton-cg - método de resolução

salescnt = salesbike[['Country', 'Customer_Gender','Age_Group', 'Sub_Category','Profit']]

X = pd.get_dummies(salescnt[['Country', 'Age_Group', 'Sub_Category','Profit']]) # transformação em variável dummy

X.columns = X.columns.str.replace(" ", "_")


y = salescnt.Customer_Gender


model.fit(X,y)

print(model.coef_)

modelo = smf.glm(formula='Age_Group ~ Profit + Country +  Sub_Category', data=salescnt, 
                 family = sm.families.Binomial()).fit()

modelo_ = print(modelo)

print(modelo.summary())

yestimativa=model.predict_proba(X)

print(confusion_matrix(y, model.predict(X))) # matriz comparação entre y (real) e model.predict(X)-predição do algoritmo ----é uma contagem de positivos e negativos

print(pd.crosstab(y, model.predict(X)))

kpis = (classification_report(y, model.predict(X))) 
print(kpis)