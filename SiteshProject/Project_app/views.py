from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Create your views here.



def Home(request):
    return render(request,'index.html')



def PulsarClassificaton(request):

    data=pd.read_csv("others\\Pulsar.csv")
    inputs=data.drop('Class','columns')
    output=data['Class']
    x_train,x_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2)
    
    model=LogisticRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    accurracy="Accuracy: "+str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100)

    submit=request.POST
    if 'submit' in submit:
        
        try:
            result=model.predict([[float(submit.get('txtmiv')),float(submit.get('txtsd')),float(submit.get('txtek')),float(submit.get('txtSkew')),float(submit.get('txtmdc')),float(submit.get('txtsdc')),float(submit.get('txtedc')),float(submit.get('txtSkewdc'))]])
           
            return render(request,'PulsarClassification.html',context={'result': 'Prediction Class: '+str(result[0]),'accuracy':accurracy})
        
        except:
            return render(request,'PulsarClassification.html',context={'result': 'Please fill all the blocks' })
        
    return render(request,'PulsarClassification.html')



def HeartAttack(request):

    data=pd.read_csv("others/heart .csv")
    inputs=data.drop('target','columns')
    outputs=data['target']
    x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)

    sc=StandardScaler()
    x_train= sc.fit_transform(x_train)
    x_test=sc.fit_transform(x_test)

    model=RandomForestClassifier()
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    Accuracy="Accuracy: "+str(accuracy_score(y_test,y_pred)*100)

    
    submit=request.POST
    if 'submit' in submit:
            
        try:
            newinputs=np.array([[int(submit.get('txt1')),int(submit.get('txt2')),int(submit.get('txt3')),int(submit.get('txt4')),int(submit.get('txt5')),int(submit.get('txt6')),int(submit.get('txt7')),int(submit.get('txt8')),int(submit.get('txt9')),float(submit.get('txt10')),int(submit.get('txt11')),int(submit.get('txt12')),int(submit.get('txt13'))]])
            newinputs=sc.transform(newinputs)
            result=model.predict(newinputs)

            if int(result[0]):
                result="Higher Chance of Heart Attack"
            else:
                result= "Lower Chance of Heart Attack"

            return render(request,'HeartAttack.html',context={'result': result,'Accuracy':Accuracy})
        
        except:
            return render(request,'HeartAttack.html',context={'result': 'Please fill all the blocks' })
    return render(request,'HeartAttack.html')