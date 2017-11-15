#Importing Libraries

import theano
import tensorflow
import keras

#Part-1 -PreProcessing the data
#-----------------------Classification(Preprocessing)-----------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Importing datasets
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

#Encoding The Categorical Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()#after dummy variable encoding remove extra column
x=x[:,1:]


#Spliting into Train_test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#---------------------The data is now pre-processed---------------------------- 

#Part-2 
#--------------------------Making the ANN Model--------------------------------

#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier=Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(input_dim=11,units=6,kernel_initializer='uniform',activation='relu'))

#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling whole ANN(i.e; applying Gradient Descent)
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#----------------Initializing the ANN(With Dropout)-------------------------------
from keras.layers import Dropout
classifier=Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(input_dim=11,units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))
#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))
#-----------------------------------------------------------------------------------

#Fitting Classifier(ANN) to train set
classifier.fit(x_train,y_train,batch_size=10,epochs=80)

#Part3
#------------------Making prediction and evaluating the model------------------
#Predicting the Results
y_pred=classifier.predict(x_test)
y_pred=(y_pred > 0.5)

#Making ConfusionMatrix(To Evaluate Performance)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#printing the OUTPUT
total=sum(sum(cm))
right_pred=cm[1,1]+(cm[0,0])
wrong_pred=cm[0,1]+(cm[1,0])
acc=((right_pred/total)*100)
print(r"Accuracy= ",acc)

#Prediciting Output for a single Observation
hw_pred=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
hw_pred=(hw_pred > 0.5)

#Part4
#-----------------Evaluating,improving and tuning the model--------------------

#Evaluating the ANN(With Dropout)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(input_dim=11,units=8,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier, batch_size=32, epochs=500)
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=1)
mean=accuracies.mean()
variance=accuracies.std()

#---------------Tuning the Model's hyperparameters------------------------
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
def build_classifier(optimizer,units):
    classifier=Sequential()
    classifier.add(Dense(input_dim=11,units=units,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=units,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,epochs=400)
parameters={'optimizer':['rmsprop','sgd'],'units':[8,14],'batch_size':[25,32]}
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring= 'accuracy', cv=10,verbose=1)
grid_search=grid_search.fit(x_train,y_train)
best_parameters=grid_search.best_params_
best_accurary=grid_search.best_score_