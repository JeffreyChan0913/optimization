from experiment import MonteCarloEstimationMinMax as MCEMM
from experiment import DistributedMCE as DMCE
from experiment import ComputeLoss as comLoss
from experiment import GradientDescent as GD
import LinearRegression as LR
import matplotlib.pyplot
from tqdm import tqdm
import pandas as pd
import time
import os

def main():
    MC_Iteration     = 80000
    numOfThread      = 10
    GD_Iteration     = 40000
    GD_alpha         = 0.001
    numOfTrial       = 100
    numOfObservation = 1000
    numOfFeatures    = 5
    testName         = '80k40k10th'
    folder           = '0604/'
    folderName       = folder + testName

    pf               = testName
    path             = folderName + 'resource/'
    numTh            = str(numOfThread)
    gdit             = str(GD_Iteration)

    try:
        os.makedirs(folderName)
    except FileExistsError:
        pass

    with open(folderName + testName + 'configuration.txt', 'w') as config:                                      # Log down the configuration
        print(f'Number of Monte Carlo iteration: {MC_Iteration}', file=config)                                  # Print the number of Monte Carlor samples to file
        print(f'Number of thread: {numOfThread}', file=config)                                                  # Print number of threads are used to file
        print(f'Gradient descent iteration: {GD_Iteration}', file=config)                                       # Print number of iteration for gradient descent to file
        print(f'Number of trail (data sets): {numOfTrial}', file=config)                                        # Print number of test run to file
        print(f'Number of observation (number of row for the generated data): {numOfObservation}', file=config) # Print the amount of row we would like to generate 
        print(f'Number of features: {numOfFeatures}', file=config)                                              # Print the amoutn of features we are simulating
        config.close()

    with open(folderName + testName + 'mc.txt', 'w') as mc,  open(folderName + testName + 'gd.txt', 'w') as gd: # Log Monte Carlo estimation and Gradient descent to file 
        for i in tqdm(range(numOfTrial)):
            xt, xts, yt, yts, beta = LR.generate_matrix(numOfObservation, numOfFeatures, True, False, False)    # Generate linear regression data

            ### Monte Carlo estimation starts here
            low, high = MCEMM(xt)                                                                               # Get the min and max as search space Monte Carlo Estimation 
            MC_train_loss, MC_beta, perf_t = DMCE(xt, yt, low, high, iteration = MC_Iteration, numberOfThreads = numOfThread)                  # Monte Carlo estimation
            MC_test_loss = comLoss(MC_beta, xts, yts)                                                           # Calculate the Mean Sqaured Error for the test set
            mc.write(','.join(str(val) for val in MC_beta.flatten().tolist()))                                  # Log Monte Carlo beta to file
            mc.write(',')                                                                                       # Write a comma to file
            mc.write(','.join(str(val) for val in beta.flatten().tolist()))                                     # Write beta (generated beta) to file
            mc.write(',')                                                                                       # Write comma to file 
            print(f'MC train loss shape: {MC_train_loss.shape}')
            output = [MC_train_loss, MC_test_loss, round(perf_t,8)]                                       # Write data to file
            mc.write(','.join(str(val) for val in output))                                                      # Write Monte Carlo training loss, Monte Carlo test loss and time to file
            mc.write(',')                                                                                       # Write comma to file
            mc.write('\n')                                                                                      # Write newline to file
            ### Monte Carlo estimation ends here

            ### Gradient Descent starts here
            GD_train_loss, GD_beta, perf_t = GD(xt, yt, alpha=GD_alpha, iteration=GD_Iteration)                 # Graident descent
            GD_train_loss = GD_train_loss[-1]                                                                   # Calculate the Mean Sqaured Error for train set
            GD_beta = GD_beta[-1]                                                                               # Get the optimal betas from gradient descent
            GD_test_loss = comLoss(GD_beta, xts, yts)                                                           # Calculate the Mean Sqaured Error for test set
            gd.write(','.join(str(val) for val in GD_beta.flatten().tolist()))                                  # Write data to file
            gd.write(',')                                                                                       # Write data to file
            gd.write(','.join(str(val) for val in beta.flatten().tolist()))                                     # Write data to file
            gd.write(',')                                                                                       # Write data to file
            output = [GD_train_loss, GD_test_loss, round(perf_t,8)]                                       # Write data to file
            gd.write(','.join(str(val) for val in output))                                                      # Write data to file
            gd.write(',')                                                                                       # Write data to file
            gd.write('\n')                                                                                      # Write data to file

            ### Gradient Descent ends here
            ### Stats
            print(f'MC TL: {MC_train_loss}\t MCTsL: {MC_test_loss}')
            print(f'GD TL: {GD_train_loss}\t GDTsL: {GD_test_loss}')
        mc.close()                                                                                              # Close Monte Carlo file
        gd.close()                                                                                              # Close Gradient Descent file

if __name__ == "__main__":
    main()
