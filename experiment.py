from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import random
import time
import sys
import os

def ComputeLoss(betas: np.array, x: np.array, y: np.array) -> np.ndarray:
	return (sum((x @ betas - y)**2))/x.shape[0]

def MonteCarloGetOptimialValues(error, beta):
    OptimalIndex = error.index(min(error))
    return error[OptimalIndex], beta[OptimalIndex] 

def MonteCarloEstiamtionAvgCol(x, n):
	return [x[:,i].sum()/n for i in range(x.shape[1])]

def MonteCarloEstimationMinMax(x):
	return min(x.min(axis=1)), max(x.max(axis=1))

def MonteCarloEstimationWorker(x, y, low, high, iteration):
    OptimalMSE = sys.maxsize
    OptimalBetas = np.nan
    pid = os.getpid()
    print(f'PID: {pid}')
    np.random.seed(seed=pid)
    for i in range(iteration):
        betas = np.round(np.random.uniform(low, high, size=(x.shape[1],1)),2)
        MSE_error = ComputeLoss(betas, x, y)
        if MSE_error < OptimalMSE:
            OptimalMSE = MSE_error
            OptimalBetas = betas
    print(f"Best MSE {OptimalMSE}")
    return OptimalBetas, OptimalMSE

def DistributedMCE(x, y, low, high, iteration= 1_000, numberOfThreads = 10):
	start = time.perf_counter()
	with ProcessPoolExecutor(max_workers=numberOfThreads) as executor:
		futures = [
			executor.submit(MonteCarloEstimationWorker, x, y, low, high, iteration)
			for _ in range(numberOfThreads)
		]
		OptimalMSE   = sys.maxsize
		OptimalBetas = np.nan 
		for future in tqdm(as_completed(futures), total=numberOfThreads):
			betas, error = future.result()
			if error <= OptimalMSE:
				OptimalMSE   = error
				OptimalBetas = betas
	stop = time.perf_counter()
	print(f"time take for DMCE {stop-start} seconds")
	return OptimalMSE, OptimalBetas, (stop-start)
	
def MonteCarloEstimation(x, y, low, high, iteration=1000):
	betasValues = []
	lossValues  = []
	a = time.process_time()
	for i in tqdm(range(iteration)):
		betas = np.random.uniform(low,high, size=(x.shape[1],1))
		betasValues.append(betas)
		error = ComputeLoss(betas, x, y)
		lossValues.append(error)
	b = time.process_time()
	print(f"Single threaded MCE took {b-a} second")
	return lossValues, betasValues

def BinaryGridSearchPointGenerator(low, high, numFeatures, direction):
	returnSize = (numFeatures,1)
	value = np.random.uniform(low, high*.33, size=returnSize) \
							  if direction == 'l' \
							  else np.random.uniform(low*.67, high, size=returnSize)
	return value
	
def BinaryGridSearch(xt, yt, low, high, iteration = 1_000):
	numFeatures     = xt.shape[1]
	numObservations = xt.shape[0]

	tempLeftValues  = np.random.uniform(low, high*.33, size=(xt.shape[1],1))
	tempRightValues = np.random.uniform(low*.67, high, size=(xt.shape[1],1))
		
	l = 0
	r = len(grid)-1
	firstRun = True
	while l <= r:
		mid = (low+high) //2
		currentLoss = ComputeLoss()
		#if firstRun:

def GradientDescent(x, y, alpha=0.0001, iteration = 1000):
	print(f"Number of iteration: {iteration}")
	start = time.perf_counter()
	betas = np.random.uniform(0,1,size=(x.shape[1],1))
	m = x.shape[0]
	betasValues = []
	betasValues.append(betas)
	lossRecord = []	
	for i in tqdm(range(iteration)):
		loss = ComputeLoss(betas, x, y)
		lossRecord.append(loss)
		betas = betas - alpha * 2/m * (x.T@(x@betas - y))
		betasValues.append(betas)
	stop = time.perf_counter()
	print(f"Gradient descent took: {stop-start} seconds")
	return lossRecord, betasValues, (stop-start)
