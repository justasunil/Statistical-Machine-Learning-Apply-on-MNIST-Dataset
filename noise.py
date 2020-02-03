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
import code

# 2nd part of the assignment

def add_Gausian_noise(data):
    
    mu = 0;
    sigma = 50;
    noise = np.random.normal(mu, sigma, [len(data),len(data[0])]) 
    return data+noise;

def Plot_data(pixels):
    pixels = pixels.reshape((28, 28))
    # Plot
    plt.figure(figsize=(3,3))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def plot_noise():
    data=[3,5,7,9,0,13,15,17,4,1];
    
    
    for i in data:
        Plot_data(train_data[i]);
    P=add_Gausian_noise(train_data);
    for i in data:
        Plot_data(P[i]);

    X=Mean(P);
    C=Covariance(P,X);
    Y=PCA(P,C,60);
    for i in data:
        Plot_data(Y[i]);

    Y=PCA(P,C,70);
    for i in data:
        Plot_data(Y[i]);

    Y=PCA(P,C,50);
    for i in data:
        Plot_data(Y[i]);
    
