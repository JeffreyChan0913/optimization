from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
from threading import Lock
from tqdm import tqdm
import numpy as np
import random
import time
import sys
import os

def ComputeLoss(betas: np.array, x: np.array, y: np.array) -> np.ndarray:
	return (np.sum((x @ betas - y)**2))/x.shape[0]

def MonteCarloGetOptimialValues(error, beta):
    OptimalIndex = error.index(min(error))
    return error[OptimalIndex], beta[OptimalIndex] 

def MonteCarloEstiamtionAvgCol(x, n):
	return [x[:,i].sum()/n for i in range(x.shape[1])]

def MonteCarloEstimationMinMax(x):
	return min(x.min(axis=1)), max(x.max(axis=1))

def MonteCarloEstimationWorker(x, y, low, high, iteration, points, lock):
    OptimalMSE = sys.maxsize
    OptimalBetas = np.nan

    t = int( time.time() * 1000.0 )
    np.random.seed( ((t & 0xff000000) >> 24) +
                    ((t & 0x00ff0000) >>  8) +
                    ((t & 0x0000ff00) <<  8) +
                    ((t & 0x000000ff) << 24)   )

    for i in tqdm(range(iteration)):
        with lock:
            while True:
                betas = np.round(np.random.uniform(low, high, size=(x.shape[1],1)),2)
                t = tuple(betas.flatten())
                if t not in points:
                    points[t] = True 
                    break
        MSE_error = ComputeLoss(betas, x, y)
        if MSE_error < OptimalMSE:
            OptimalMSE = MSE_error
            OptimalBetas = betas
    return OptimalBetas, OptimalMSE

def DistributedMCE(x, y, low, high, iteration= 1_000, numberOfThreads = 10):
	start = time.perf_counter()
	with Manager() as bigBoyManager:
		points = bigBoyManager.dict()
		lock   = bigBoyManager.Lock() 
		with ProcessPoolExecutor(max_workers=numberOfThreads) as executor:
			futures = [
                executor.submit(MonteCarloEstimationWorker, x, y, low, high, iteration, points, lock)
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
