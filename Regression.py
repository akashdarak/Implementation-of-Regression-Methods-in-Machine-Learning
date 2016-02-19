import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def Lasso(xTrain,yTrain):
    lambda1bda1 = 0.01
    lambda1bda2 = lambda1bda1
    
    #calculating the number of rows in w
    rows=0
    maxlambda1bda = 10000000000
    while(lambda1bda2<maxlambda1bda):
        rows+=1
        lambda1bda2=lambda1bda2*10
        
    w=np.zeros([rows,len(xTrain[0])])
    
    #Training
    i=0
    alllambda1 =[]
    
    while (lambda1bda1<maxlambda1bda):
        clf = linear_model.Lasso(alpha=lambda1bda1)
        clf.fit(xTrain,yTrain)
        w[i,] =    clf.coef_     
        alllambda1.append(lambda1bda1)
        lambda1bda1=lambda1bda1*10
        i+=1
    return(w,alllambda1)    

def Data_Acq() :
    data=np.genfromtxt("ship_speed_fuel.csv", delimiter=",")
    data=np.array(data)
    data=np.delete(data,len(data[0]),0)
    
    print(data)
    k=np.zeros(len(data))
    for i in range(0,len(data[0])):
        for j in range(0,len(data)):
            k[j]=data[j][i]
        mean=np.mean(k)             #Mean
        std=np.std(k)               #Standard Deviation
        for j in range(0,len(data)):
            data[j][i]=(data[j][i]-mean)/std
    
    print(data)
    xTrainLen = round(len(data)*0.85)
    y=data[:,len(data[0])-1]
    ytest=y[0:xTrainLen]
    ytrain=y[xTrainLen:]
    
    XData=np.delete(data,len(data[0])-1,1)
    Xtest=XData[0:xTrainLen,:]
    Xtrain=XData[xTrainLen:,:]
    
    print ("Xtest", Xtest)
    print ("Xtrain", Xtrain)
    return(Xtrain,ytrain,Xtest,ytest)
    
def Data_Acq2():
    data=np.genfromtxt("ship_speed_fuel.csv", delimiter=",", comments="#")
    y= data[1:,len(data[0])-1]
    x= data[1:,:(len(data[0])-1)]
    
    #Standardization of x
    for i in range(len(x[0])):
        mean = np.mean(x[:,i])
        stan = np.std(x[:,i])
        x[:,i]=(x[:,i]-mean)/stan 
    
    q=np.ones((len(x),len(x[0])+1))
    q[:,1:]=x
    x=q
    
    xTrainLen = round(len(x)*0.66)
    xTrain = x[1:(xTrainLen-1),:]
    yTrain = y[1:(xTrainLen-1)]
    
    xTest = x[xTrainLen:,:]
    yTest = y[xTrainLen:]
    return(xTrain,yTrain,xTest,yTest)
   
def MAD1(w,xTest,yTest,alllambda1):
    mad = []
    for i in range(len(w)):
        errorSum = 0
        for j in range(len(xTest)):
            errorSum += abs(yTest[j] - np.dot(w[i,],xTest[j,]))
        mad.append(errorSum/(len(xTest)))
    plt.plot(alllambda1,mad,'r-',label = "lasso")
    plt.xscale('log')
    plt.ylabel("MAD")
    plt.xlabel("lambda1bda")
    plt.title("red-Wine")
    plt.legend(loc = 4)
    
def MAD2(w,xTest,yTest,alllambda1):
    mad = []
    for i in range(len(w)):
        errorSum = 0
        for j in range(len(xTest)):
            errorSum += abs(yTest[j] - np.dot(w[i,],xTest[j,]))
        mad.append(errorSum/(len(xTest)))
    plt.plot(alllambda1,mad,'b-',label = "elastic")
    plt.xscale('log')
    plt.ylabel("MAD")
    plt.xlabel("lambda1bda")
    plt.title("red-Wine")
    plt.legend(loc = 4)
    
def MAD3(w,xTest,yTest,alllambda1):
    mad = []
    for i in range(len(w)):
        errorSum = 0
        for j in range(len(xTest)):
            errorSum += abs(yTest[j] - np.dot(w[i,],xTest[j,]))
        mad.append(errorSum/(len(xTest)))
    print("MAD:",mad)
    plt.plot(alllambda1,mad,'g-',label = "Ridge")
    plt.xscale('log')
    plt.ylabel("MAD")
    plt.xlabel("lambda1bda")
    plt.title("red-Wine")
    plt.legend(loc = 4)

def ElasticNet(Xtrain,ytrain):
    z=0    
    rows=0
    lambda1 = 0.0001
    lambda2 = lambda1
    finallambda =[]
    maxlambda = 10000000
    while(lambda2<maxlambda):
        rows = rows + 1
        lambda2=10*lambda2
        
    w=np.zeros([rows,len(Xtrain[0])])
    while (lambda1<maxlambda):
        clf = linear_model.ElasticNet(alpha=lambda1)
        clf.fit(Xtrain,ytrain)
        w[z,] =    clf.coef_     
        finallambda.append(lambda1)
        lambda1=lambda1*10
        z = z + 1
    return(w,finallambda)
    
def Ridge(xTrain,yTrain):
    lambda1bda1 = 0.01
    lambda1bda2 = lambda1bda1
    
    #calculating the number of rows in w
    rows=0
    maxlambda1bda = 10000000000
    while(lambda1bda2<maxlambda1bda):
        rows+=1
        lambda1bda2=lambda1bda2*10
        
    w=np.zeros([rows,len(xTrain[0])])
    
    #Training
    i=0
    alllambda1 =[]
    iden = np.identity(len(xTrain[0]))
    #iden[0][0]=0
    
    while (lambda1bda1<maxlambda1bda):
        w[i,]=(np.linalg.solve(np.dot(xTrain.T,xTrain) + lambda1bda1 * iden, np.dot(xTrain.T,yTrain)))
        alllambda1.append(lambda1bda1)
        lambda1bda1=lambda1bda1*10
        i+=1
    return(w,alllambda1) 
        
if __name__=='__main__' :
    xTrain,yTrain,xTest,yTest = Data_Acq2()
    w,alllambda1=Lasso(xTrain,yTrain)
    MAD1(w,xTest,yTest,alllambda1)
    w,alllambda1=ElasticNet(xTrain,yTrain)
    MAD2(w,xTest,yTest,alllambda1)
    
    w,alllambda1=Ridge(xTrain,yTrain)
    MAD3(w,xTest,yTest,alllambda1)
    plt.show()
