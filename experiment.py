import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from tqdm import tqdm

def ComputeLoss(betas: np.array, xt: int, yt: int) -> np.ndarray:
	return sum((xt @ betas - yt)**2)

def MonteCarloGetOptimialValues(error, beta):
    OptimalIndex = error.index(min(error))
    return error[OptimalIndex], beta[OptimalIndex] 

def MonteCarloEstiamtionAvgCol(xt, n):
	return [xt[:,i].sum()/n for i in range(xt.shape[1])]

def MonteCarloEstimationMinMax(xt):
	return min(xt.min(axis=1)), max(xt.max(axis=1))

def MonteCarloEstimationWorker(xt, yt, low, high, iteration):
    OptimalMSE = sys.maxsize
    for i in range(iteration):
        betas = np.round(np.random.uniform(low, high, size=(xt.shape[1],1)),2)
        MSE_error = sum(ComputeLoss(betas, xt, yt))/xt.shape[0]
        OptimalBetas = np.nan
        if MSE_error < OptimalMSE:
            OptimalError = MSE_error
            OptimalBetas = betas
    return OptimalBetas, OptimalMSE

def DistributedMCE(xt, yt, iteration= 1_000, numberOfThreads = 10):
	low, high = MonteCarloEstimationMinMax(xt)
	print("Starting up the threads")
	with ThreadPoolExecutor(max_workers=numberOfThreads) as executor:
		futures = [
			executor.submit(MonteCarloEstimationWorker, xt, yt, low, high, iteration)
			for _ in range(numberOfThreads)
		]
		print("All tasks assigned to threads")
		OptimalMSE   = sys.maxsize
		OptimalBetas = np.nan 
		print("Pending results from threads")
		for future in tqdm(as_completed(futures), total=numberOfThreads):
			betas, error = future.result()
			if error <= OptimalMSE:
				OptimalMSE   = error
				OptimalBetas = betas
		
	print("---------------------- Returning results -----------------------") 
	return OptimalMSE, OptimalBetas
	
def MonteCarloEstimation(xt, yt, iteration=500):
	low, high = MonteCarloEstimationMinMax(xt)
	betasValues = []
	lossValues  = []
	for i in tqdm(range(1000)):
		betas = np.random.uniform(low,high, size=(xt.shape[1],1))
		betasValues.append(betas)
		error = ComputeLoss(betas, xt, yt)
		lossValues.append(error)
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

def GradientDescent(xt, yt, alpha=0.0001, iteration = 1000):
	betas = np.random.uniform(0,1,size=(xt.shape[1],1))
	m = xt.shape[0]
	betasValues = []
	betasValues.append(betas)
	lossRecord = []
	
	for i in tqdm(range(iteration)):
		loss = ComputeLoss(betas, xt, yt)
		lossRecord.append(loss)
		betas = betas - alpha * 2/m * (xt.T@(xt@betas - yt))
		betasValues.append(betas)

	return lossRecord, betasValues
