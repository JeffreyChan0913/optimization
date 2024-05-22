from experiment import DistributedMCE as DMCE
from experiment import MonteCarloEstimationMinMax as MCEMM
from experiment import ComputeLoss as comLoss
from experiment import GradientDescent as GD
import LinearRegression as LR
from tqdm import tqdm
import time

MC_Iteration     = 1000
numOfThread      = 10
GD_Iteration     = MC_Iteration*numOfThread
GD_alpha         = 0.001
numOfTrial       = 10
numOfObservation = 1000

with open('0521v1/configuration.txt', 'w') as config:                                                       # Log down the configuration
    print(f'Number of Monte Carlo iteration: {MC_Iteration}', file=config)                                  # Print the number of Monte Carlor samples to file
    print(f'Number of thread: {numOfThread}', file=config)                                                  # Print number of threads are used to file
    print(f'Gradient descent iteration: {GD_Iteration}', file=config)                                       # Print number of iteration for gradient descent to file
    print(f'Number of trail (data sets): {numOfTrial}', file=config)                                        # Print number of test run to file
    print(f'Number of observation (number of row for the generated data): {numOfObservation}', file=config) # Print the amount of row we would like to generate 
    config.close()

with open('0522v1/mc.txt', 'w') as mc,  open('0522v1/gd.txt', 'w') as gd:                                   # Log Monte Carlo estimation and Gradient descent to file 
    for i in tqdm(range(numOfTrial)):
        xt, xts, yt, yts, beta = LR.generate_matrix(numOfObservation, 3, True)                              # Generate linear regression data
        
        ### Monte Carlo estimation starts here
        low, high = MCEMM(xt)                                                                               # Get the min and max as search space Monte Carlo Estimation 
        
        a = time.process_time()                                                                             # Performance metric: track the time
        MC_train_loss, MC_beta = DMCE(xt, yt, low, high, MC_Iteration)                                      # Monte Carlo estimation
        b = time.process_time()                                                                             # Performance metric: track the time 
        total = b - a                                                                                       # Get the total time used
        MC_test_loss = comLoss(MC_beta, xts, yts)                                                           # Calculate the Mean Sqaured Error for the test set
        mc.write(','.join(str(val) for val in MC_beta.flatten().tolist()))                                  # Log Monte Carlo beta to file
        mc.write(',')                                                                                       # Write a comma to file
        mc.write(','.join(str(val) for val in beta))                                                        # Write beta (generated beta) to file
        mc.write(',')                                                                                       # Write comma to file 
        output = [MC_train_loss[0], MC_test_loss[0], round(total,8)]                                              # Write data to file
        mc.write(','.join(str(val) for val in output))                                                      # Write Monte Carlo training loss, Monte Carlo test loss and time to file
        mc.write(',')                                                                                       # Write comma to file
        mc.write('\n')                                                                                      # Write newline to file
        ### Monte Carlo estimation ends here

        ### Gradient Descent starts here
        a = time.process_time()                                                                             # Performance metric: track the time
        GD_train_loss, GD_beta = GD(xt, yt, alpha=0.0001, iteration=GD_Iteration)                           # Graident descent
        b = time.process_time()                                                                             # Performance metric: track the time
        total = b - a                                                                                       # Performance metric: track the time
        GD_train_loss = GD_train_loss[-1]                                                                   # Calculate the Mean Sqaured Error for train set
        GD_beta = GD_beta[-1]                                                                               # Get the optimal betas from gradient descent
        GD_test_loss = comLoss(GD_beta, xts, yts)                                                           # Calculate the Mean Sqaured Error for test set
        gd.write(','.join(str(val) for val in GD_beta.flatten().tolist()))                                  # Write data to file
        gd.write(',')                                                                                       # Write data to file
        gd.write(','.join(str(val) for val in beta))                                                        # Write data to file
        gd.write(',')                                                                                       # Write data to file
        output = [GD_train_loss[0], GD_test_loss[0], round(total,8)]                                        # Write data to file
        gd.write(','.join(str(val) for val in output))                                                      # Write data to file
        gd.write(',')                                                                                       # Write data to file
        gd.write('\n')                                                                                      # Write data to file
        print('Actual betas: {}'.format(beta))                                                              # Print beta 
        print('MC betas:     {}'.format(MC_beta))                                                           # Print beta hat (Monte Carlo) 
        print('GD betas:     {}'.format(GD_beta))                                                           # Print beta hat (Gradient descent)
        ### Gradient Descent ends here

    mc.close()                                                                                              # Close Monte Carlo file
    gd.close()                                                                                              # Close Gradient Descent file
