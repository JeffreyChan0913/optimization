import matplotlib.pyplot as plt
import pandas as pd
import os

pf      = '80k40k10th'
folder  = '0604/'
path    = folder + pf + '/resource/'
gdit    = '40k'
numTh   = '10'
try:
    os.makedirs(path)
except FileExistsError:
    pass

mccol = ['beta0','beta1','beta2','beta3','beta4','beta5','b0','b1','b2','b3','b4','b5','MCtrainloss','MCtestloss','sec']
gdcol = ['beta0','beta1','beta2','beta3','beta4','beta5','b0','b1','b2','b3','b4','b5','GDtrainloss','GDtestloss','sec']
mc1 = pd.read_csv(folder + pf + '/' + pf+'mc.txt', names=mccol, header=None, index_col=False)
gd1 = pd.read_csv(folder + pf + '/' + pf+'gd.txt', names=gdcol, header=None, index_col=False)

plt.plot(range(1,101), mc1.MCtestloss, label='Monte Carlo')
plt.plot(range(1,101), gd1.GDtestloss, label='Gradient Descent')
plt.xlabel('Data Set')
plt.ylabel('MSE')
plt.title('MSE on Each Test Dataset')
plt.legend()
plt.savefig(path + pf + 'mc1_gd1_tsl.png',dpi=720, bbox_inches='tight')
plt.close()

plt.plot(range(1,101), mc1.MCtrainloss, label='Monte Carlo')
plt.plot(range(1,101), gd1.GDtrainloss, label='Gradient Descent')
plt.xlabel('Data Set')
plt.ylabel('MSE')
plt.title('MSE on Each Train Dataset')
plt.legend()
plt.savefig(path + pf + 'mc1_gd1_tl.png', dpi=720)
plt.close()

plt.plot(range(1,101), mc1.sec, label='Monte Carlo')
plt.plot(range(1,101), gd1.sec, label='Gradient Descent')
plt.legend(loc='center')
plt.title('Time Used for Each Dataset')
plt.ylabel('Second')
plt.xlabel('Dataset')
plt.savefig(path + pf + 'mc1_gd1_time.png',dpi=720)
plt.ylim((0,5))
plt.close()

plt.plot(range(1,101), gd1.sec, label='Gradient Descent')
plt.xlabel('Data Set')
plt.ylabel('Second')
plt.title('Gradient Descent (' + gdit +' Iterations) Performance on Each Dataset')
plt.axhline(y=gd1.sec.mean(), color='darkorange', linestyle='-')
plt.legend()
plt.text(7.5,0.04101, 'mean: ' + str(round(gd1.sec.mean(),5)) + ' secs') 
plt.savefig(path + pf + 'gd1_time.png', dpi=720)
plt.close()

plt.plot(range(1,101), mc1.sec, label='Monte Carlo')
plt.xlabel('Data Set')
plt.ylabel('Second')
plt.title('Monte Carlo Multi Threaded (' + numTh + ') Performance on Each Dataset')
plt.axhline(y=mc1.sec.mean(), color='darkorange', linestyle='-')
plt.legend()
plt.text(7.5,0.04101, 'mean: ' + str(round(mc1.sec.mean(),5)) + ' secs') 
plt.savefig(path + pf + 'mc1_time.png', dpi=720)
plt.close()

plt.plot(range(1,101), gd1.GDtrainloss, label='Train')
plt.plot(range(1,101), gd1.GDtestloss, label='Test')
plt.xlabel('Data Set')
plt.ylabel('MSE')
plt.title('Graident Descent MSE Train vs Test on Each Dataset')
plt.legend()
plt.savefig(path + pf + 'gd1_tl_tsl.png', dpi=720)
plt.close()

plt.plot(range(1,101), mc1.MCtrainloss, label='Train')
plt.plot(range(1,101), mc1.MCtestloss, label='Test')
plt.xlabel('Data Set')
plt.ylabel('MSE')
plt.title('Monte Carlo MSE Train vs Test on Each Dataset')
plt.legend()
plt.savefig(path + pf + 'mc_tl_tsl.png', dpi=720, bbox_inches='tight')
plt.close()
