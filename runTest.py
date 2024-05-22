from experiment import DistributedMCE as DMCE
from experiment import ComputeLoss as comLoss
from experiment import GradientDescent as GD
import LinearRegression as LR
from tqdm import tqdm
import time

MC_Iteration = 1000
numOfThread  = 10
GD_Iteration = MC_Iteration*numOfThread
GD_alpha     = 0.000001
numOfTrial   = 100
numOfObservation = 1000

with open('0521v1/configuration.txt', 'w') as config:
    print(f'Number of Monte Carlo iteration: {MC_Iteration}', file=config)
    print(f'Number of thread: {numOfThread}', file=config)
    print(f'Gradient descent iteration: {GD_Iteration}', file=config)
    print(f'Number of trail (data sets): {numOfTrial}', file=config)
    print(f'Number of observation (number of row for the generated data): {numOfObservation}', file=config)
    config.close()

with open('0521v1/mc.txt', 'w') as mc,  open('0521v1/gd.txt', 'w') as gd:
    for i in tqdm(range(numOfTrial)):
        xt, xts, yt, yts, beta = LR.generate_matrix(numOfObservation, 3, True)
        
        a = time.process_time()
        MC_train_loss, MC_beta = DMCE(xt, yt, MC_Iteration)
        b = time.process_time() 
        total = b - a
        MC_train_loss = MC_train_loss / xt.shape[0]
        MC_test_loss = sum(comLoss(MC_beta, xts, yts))/xts.shape[0]
        mc.write(','.join(str(val) for val in MC_beta.flatten().tolist()))
        mc.write(',')
        mc.write(','.join(str(val) for val in beta)) 
        mc.write(',')
        output = [MC_train_loss, MC_test_loss, round(total,8)]
        mc.write(','.join(str(val) for val in output))
        mc.write(',')
        mc.write('\n')
        a = time.process_time()
        GD_train_loss, GD_beta = GD(xt, yt, alpha=0.0001, iteration=GD_Iteration)
        b = time.process_time()
        total = b - a
        GD_test_loss = sum(comLoss(GD_beta, xts, yts))/xts.shape[0]
        GD_beta = GD_beta[-1]
        GD_train_loss = GD_train_loss[-1]/xt.shape[0]
        gd.write(','.join(str(val) for val in GD_beta.flatten().tolist()))
        gd.write(',')
        gd.write(','.join(str(val) for val in beta)) 
        gd.write(',')
        output = [GD_train_loss[0], GD_test_loss[0], round(total,8)]
        gd.write(','.join(str(val) for val in output))
        gd.write(',')
        gd.write('\n')
        print('Actual betas: {}'.format(beta))
        print('MC betas:     {}'.format(MC_beta))
        print('GD betas:     {}'.format(GD_beta))

    mc.close()
    gd.close()
