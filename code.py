import idx2numpy
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy as geek
import matplotlib
import scipy.linalg
import pandas as pd
import pickle

# Reading data from idx format
train_data = idx2numpy.convert_from_file('dir(dataset)/train-images-idx3-ubyte')

train_label = idx2numpy.convert_from_file('dir(dataset)/MNIST/train-labels-idx1-ubyte')

test_data = idx2numpy.convert_from_file('dir(dataset)/MNIST/t10k-images-idx3-ubyte')

test_label = idx2numpy.convert_from_file('dir(dataset)/MNIST/t10k-labels-idx1-ubyte')


"""
#------------for test dataset
with open('test_data','rb') as f:
    test_data=pickle.load(f)

#------------for train dataset

with open('train_data','rb') as f:
    train_data=pickle.load(f)

"""
def Mean(data):
    size=len(data);
    dim=len(data[0]);
    X=geek.zeros([dim,1]);
    
    for j in range(len(data)):
        Y=data[j];
        Y.shape=(dim,1);
        #X+=Y;
        X=np.add(X, Y, out=X, casting="unsafe");
        
    X=np.divide(X,size);
    print("success");
    return X;

def Covariance(data,Xmean):
    size=len(data);
    dim=len(data[0]);
    C=geek.zeros([dim,dim]);
    for j in range(len(data)):
        Y=data[j];
        Y.shape=(dim,1);
        Temp=Y-Xmean;
        #C+=(Temp.dot(Temp.transpose()));
        C=np.add(C, Temp.dot(Temp.transpose()), out=C, casting="unsafe");
    C=np.divide(C,(size-1));
    print("success");
    return C;


def Mean_by_class(data,w,label):
    L=[];
    N=0;
    for i in range(len(data)):
        if label[i]==w:
            L.append(data[i]);
            N+=1;
    Y=Mean(L);
    print("success");
    return L , Y , N;


def Covariance_by_class(data,w,label):
    
    X , Y , N = Mean_by_class(data,w,label);
    
    C=Covariance(X,Y);
    print("success");
    return C , Y , N;

def find_num_pc(e,eigen):
    total=0;
    for i in range(len(eigen)):
        total+=eigen[i]

    j=0;
    expect=0;
    while(j<len(eigen) and (expect/total)*100 < e):
        expect+=eigen[j]
        j+=1;
    
    print("success");
    return j;

# find PCA for MNIST dataset

def PCA(data,covariance,energy):
    
    #find eigenvalue and eigenvector
    eigvalues, eigvectors = la.eig(covariance)
    

    # find p largest eigenvalue
    idx = eigvalues.argsort()[::-1]   
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:,idx]
    # reduce to p dimension from 784
    p=find_num_pc(energy,eigvalues)
    eigvectors = eigvectors[:,:p]
    eigv_tran=np.array(eigvectors).transpose()

    Y=eigv_tran.dot(np.array(data).transpose());
    """#retrieve image after PCA
    P=np.linalg.pinv(eigvectors.dot(eigv_tran))
    S=P.dot(eigvectors.dot(Y))
    S_tran=S.transpose()"""
    
    print("success PCA");
    return Y , eigv_tran;
    


def Plot(Y):
    # naming the x axis 
    plt.xlabel('x - axis') 
    # naming the y axis 
    plt.ylabel('y - axis') 
      
    # giving a title to my graph 
    plt.title('PCA')


    #plot the value after applying PCA

    #plt.plot(Y[0], Y[1])
    x=list(Y[0])
    y=list(Y[1])
    colors = ['red','green','blue','purple','black','yellow','pink','orange']
    plt.scatter(x, y, c=train_label, cmap=matplotlib.colors.ListedColormap(colors))
      
    # function to show the plot 
    plt.show()

            
            

# find FDA for MNIST data
def FDA(data,covariance,cov): #you can add energy
    
    Sw = cov[0] + cov[1] + cov[2] + cov[3] + cov[4] + cov[5] + cov[6] + cov[7] + cov[8] + cov[9];

    Sb = covariance - Sw;

    Sw_inv = np.linalg.pinv(Sw);

    Ch_eq = Sw_inv.dot(Sb);

    #find eigenvalue and eigenvector of characteristic equation

    eigvalues , eigvectors = la.eig(Ch_eq);
    
    # find p largest eigenvalue
    idx = eigvalues.argsort()[::-1]   
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:,idx]
    
    # reduce to p dimension from 784
    #p=find_num_pc(energy,eigvalues)
    eigvectors = eigvectors[:,:2] # there are 10 classes
    
    eigv_tran=np.array(eigvectors).transpose()

    Y=eigv_tran.dot(np.array(data).transpose());
    print("succes FDA");
    return Y , eigv_tran;



# Find LDA

def Attributes(data,label):
    cov=[]
    mean=[]
    P=geek.zeros([10,1]);
    N=geek.zeros([10,1]);
    
        
    cov0 , mean0 , N[0]= Covariance_by_class(data,0,label);
    cov1 , mean1 , N[1]= Covariance_by_class(data,1,label);
    cov2 , mean2 , N[2]= Covariance_by_class(data,2,label);
    cov3 , mean3 , N[3]= Covariance_by_class(data,3,label);
    cov4 , mean4 , N[4]= Covariance_by_class(data,4,label);
    cov5 , mean5 , N[5]= Covariance_by_class(data,5,label);
    cov6 , mean6 , N[6]= Covariance_by_class(data,6,label);
    cov7 , mean7 , N[7]= Covariance_by_class(data,7,label);
    cov8 , mean8 , N[8]= Covariance_by_class(data,8,label);
    cov9 , mean9 , N[9]= Covariance_by_class(data,9,label);

    cov.append(cov0)
    cov.append(cov1)
    cov.append(cov2)
    cov.append(cov3)
    cov.append(cov4)
    cov.append(cov5)
    cov.append(cov6)
    cov.append(cov7)
    cov.append(cov8)
    cov.append(cov9)

    mean.append(mean0)
    mean.append(mean1)
    mean.append(mean2)
    mean.append(mean3)
    mean.append(mean4)
    mean.append(mean5)
    mean.append(mean6)
    mean.append(mean7)
    mean.append(mean8)
    mean.append(mean9)
    
    #prior probability of all classes from 0-9
    for i in range(len(P)):
        P[i]=N[i]/len(data);
    
    return cov , mean , P ;

#= Attributes(train_data,train_label); cov , mean , P ;

def LDA(S,mean,cov,P):

        
    g=geek.zeros([10,1])
    g[0]=-(1/2)*(((S-mean[0]).transpose()).dot((np.linalg.pinv(cov[0])).dot(S-mean[0]))) + np.log(P[0]);
    g[1]=-(1/2)*(((S-mean[1]).transpose()).dot((np.linalg.pinv(cov[1])).dot(S-mean[1]))) + np.log(P[1]);
    g[2]=-(1/2)*(((S-mean[2]).transpose()).dot((np.linalg.pinv(cov[2])).dot(S-mean[2]))) + np.log(P[2]);
    g[3]=-(1/2)*(((S-mean[3]).transpose()).dot((np.linalg.pinv(cov[3])).dot(S-mean[3]))) + np.log(P[3]);
    g[4]=-(1/2)*(((S-mean[4]).transpose()).dot((np.linalg.pinv(cov[4])).dot(S-mean[4]))) + np.log(P[4]);
    g[5]=-(1/2)*(((S-mean[5]).transpose()).dot((np.linalg.pinv(cov[5])).dot(S-mean[5]))) + np.log(P[5]);
    g[6]=-(1/2)*(((S-mean[6]).transpose()).dot((np.linalg.pinv(cov[6])).dot(S-mean[6]))) + np.log(P[6]);
    g[7]=-(1/2)*(((S-mean[7]).transpose()).dot((np.linalg.pinv(cov[7])).dot(S-mean[7]))) + np.log(P[7]);
    g[8]=-(1/2)*(((S-mean[8]).transpose()).dot((np.linalg.pinv(cov[8])).dot(S-mean[8]))) + np.log(P[8]);
    g[9]=-(1/2)*(((S-mean[9]).transpose()).dot((np.linalg.pinv(cov[9])).dot(S-mean[9]))) + np.log(P[9]);

    min=0;
    for i in range(1,10):
        if(g[i]>g[min]):
            min=i;
        else:
            continue;
    return min;



def LDA_Accuracy(test,low,high,test_label,train,train_label):
    test=np.array(test);
    cov , mean , P = Attributes(train,train_label);
    
    c=0;
    for i in range(low,high):
        S=np.array(test[i]);
        S.shape=(len(S),1);
        if(LDA(S,mean,cov,P)==test_label[i]):
            c+=1;

    return str((c/(high-low))*100) + "%";



#print(LDA_Accuracy(5,15))


def LDA_after_PCA(energy,low,high):
    
    X=Mean(train_data);
    C=Covariance(train_data,X);
    Y , eigv_tran=PCA(train_data,C,energy);
    
    test=np.array(eigv_tran.dot(np.array(test_data).transpose())).transpose();
    print("successL");
    print(LDA_Accuracy(test,low,high,test_label,Y.transpose(),train_label));



def LDA_after_FDA(low,high):
    X=Mean(train_data);
    C=Covariance(train_data,X);
    cov , mean , P = Attributes(train_data,train_label);
    Y , eigv_tran= FDA(train_data,C,cov);
    
    test=np.array(eigv_tran.dot(np.array(test_data).transpose())).transpose();
    print("successL");
    print(LDA_Accuracy(test,low,high,test_label,Y.transpose(),train_label));

def LDA_after_FDA_after_PCA(low,high,energy):
    #apply PCA
    X=Mean(train_data);
    C=Covariance(train_data,X);
    Y , eigv_tran=PCA(train_data,C,energy);
    #apply FDA
    XF=Mean(Y.transpose());
    CF=Covariance(Y.transpose(),XF);
    cov , mean , P = Attributes(Y.transpose(),train_label);
    YF , eigvF_tran = FDA(Y.transpose(),CF,cov);
    
    test=np.array(eigv_tran.dot(np.array(test_data).transpose())).transpose();
    testF=np.array(eigvF_tran.dot(np.array(test).transpose())).transpose();
    print("successL");
    print(LDA_Accuracy(testF,low,high,test_label,YF.transpose(),train_label));
    
