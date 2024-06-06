import numpy as np
import time

def beta_coefs(x, y) -> np.array:
    return np.linalg.inv(x.T@x) @ x.T @ y

def getMu():
    upper_bound = np.random.randint(0,20)
    return np.round(np.random.uniform(-upper_bound, upper_bound), 2)

def getSd():
    return np.round(np.random.uniform(0, np.random.randint(0,6)), 2)

def info() -> None:
	print('--------------------------- Input Configuration  ----------------------------') 
	print('Number of data point:             {}'.format(row))
	print('Number of features:               {}'.format(col))
	print('With intercept:                   {}'.format(intercept))
	print('With more test bias:              {}'.format(moreBais))
	print('---------------------------- Data Configuration  ----------------------------') 
	print('Generated x-train data dimension: {}'.format(xt.shape))
	print('Generated x-test data dimension:  {}'.format(xts.shape))
	print('Generated y-train data dimension: {}'.format(yt.shape))
	print('Generated y-test data dimension:  {}'.format(yts.shape))
	print('Generated train epsilon dimension:{}'.format(epsilon_train.shape))
	print('Generated test epsilon dimension: {}'.format(epsilon_test.shape))
	print('Generated beta dimension:         {}'.format(betas.shape))
	print('Generated betas:                  {}'.format(betas))
	print('------------------------------------------------------------------------------') 


def generate_matrix(row:int, col:int, intercept:bool, moreBais: bool, verbose = False) -> np.array:

	t = int(time.time() * 1000.0)
	np.random.seed( ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)   )
				
	mu = getMu()
	sd = getSd()

	test_data_cutoff = int(row * 0.3)

	if intercept:
		xt = np.ones(row)
		xts = np.ones(test_data_cutoff)
		betas = np.random.uniform(-50,50, 1).reshape(-1,1)

	bound = np.random.randint(0,20)
	betas = np.concatenate((betas, np.random.uniform(-bound, bound, col).reshape(-1,1)))
	
	for i in range(col):
		mu = getMu()
		sd = getSd()
		xt  = np.column_stack((xt, np.random.normal(mu,sd,row)))
		if moreBais:
			mu = getMu()
			sd = getSd()
		xts = np.column_stack((xts, np.random.normal(mu,sd,test_data_cutoff)))
				
	mu = getMu()
	sd = getSd()
	epsilon_train = np.array(np.random.normal(mu,sd,row)).reshape(-1,1)
	epsilon_test  = np.array(np.random.normal(mu,sd, test_data_cutoff)).reshape(-1,1)
	yt            = xt @ betas + epsilon_train
	yts           = xts @ betas + epsilon_test 
	
	if verbose:	
		info()
	return xt, xts, yt, yts, betas
