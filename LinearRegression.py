import numpy as np
import random
from numpy.linalg import inv

def beta_coefs(x, y):
    return np.linalg.inv(x.T@x) @ x.T @ y

def generate_matrix(row:int, col:int, intercept:bool) -> np.array:

    xt      = []
    xts     = []
    yt      = []
    yts     = []
    betas   = []
    test_data_cutoff = int(row * 0.3)
    if intercept:
        xt.append(np.ones(row))
        xts.append(np.ones(test_data_cutoff))
        
    for i in range(col):
        mu = round(random.uniform(-10,10),2)
        sd = round(random.uniform(0,6),2)
        xt.append(np.random.normal(mu,sd,row))
        mu = round(random.uniform(-10,10),2)
        sd = round(random.uniform(0,6),2)    
        xts.append(np.random.normal(mu,sd,test_data_cutoff))
                
    mu = round(random.uniform(-10,10),2)
    sd = round(random.uniform(0,6),2) 
    epsilon = np.random.normal(mu,sd,row)

    betas.append(random.uniform(-200,200))
    for i in range(col):
        betas.append(round(random.uniform(-10,10),2))
    
    xt      = np.array(xt).T
    xts     = np.array(xts).T
   
    print('--------------------------- Input Configuration  ----------------------------') 
    print('Number of data point:             {}'.format(row))
    print('Number of features:               {}'.format(col))
    print('With intercept:                   {}'.format(intercept))
    print('---------------------------- Data Configuration  ----------------------------') 
    print('Generated x-train data dimension: {}'.format(xt.shape))
    print('Generated x-test data dimension:  {}'.format(xts.shape))
    print('Generated epsilon length:         {}'.format(len(epsilon)))
    print('Generated betas:                  {}'.format(betas))
    #print('Generated equation:               {}'.format(equation(betas, intercept)))
    print('------------------------------------------------------------------------------') 

    for i in range(row):
        sum = epsilon[i]
        for j in range(col):    
            sum += betas[j]*xt[i,j:j+1]
        yt.append(sum) 
    
    for i in range(test_data_cutoff):
        sum = epsilon[i]
        for j in range(col):
            sum += betas[j]*xt[i,j:j+1]
        yts.append(sum)
    
    return xt, xts, yt, yts, betas

