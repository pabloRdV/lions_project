import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

def examples_CMNIST(data):
  nclasses = len(torch.unique(data['labels']))
  for i in range(nclasses):
    print('Class',i)
    idx_class = data['labels']==i
    nsamples_class = idx_class.sum()
    print('  Number of samples:', nsamples_class)
    sel = np.random.randint(0,nsamples_class,size=10)
    images = data['images'][idx_class][sel] #select 10 rand images from class i
    grid = torchvision.utils.make_grid(images, nrow=5, padding=10)
    grid = np.moveaxis(grid.numpy(), 0, 2) # move channels to last dim
    print('  Labels:', data['labels'][idx_class][sel])
    print('  Colors:', data['colors'][idx_class][sel])
    agreement = 100.*(data['colors'][idx_class]==data['labels'][idx_class]).sum()/len(data['colors'][idx_class])
    print('  Agreement colors/labels (accross the whole class): {:.3f}%'.format(agreement))
    plt.imshow(grid)
    plt.show()

def display_confs(confs):
  num_subgroups = len(confs)
  fig, axs = plt.subplots(1,num_subgroups,figsize=(15,6))
  for i,ax in enumerate(axs):
    disp = ConfusionMatrixDisplay(confs[i])
    disp.plot(ax=ax)
    ax.set_title('Confusion Matrix - Subgroup {}'.format(i))
  plt.show()


def plot_subgroup_losses(epoch_loss, epoch_mi, epoch_loss_adv, runs=list(range(10)), loss_name='Demographic Parity Gap'):
  fig, axs = plt.subplots(1,3,figsize=(21,6))
  titles = [loss_name, 'Mutual Information (unweighted)', 'CE loss (adversary)']

  results_avg = [epoch_loss[runs,:].mean(axis=0), epoch_mi[runs,:].mean(axis=0), epoch_loss_adv[runs,:].mean(axis=0)]
  results_std = [epoch_loss[runs,:].std(axis=0), epoch_mi[runs,:].std(axis=0), epoch_loss_adv[runs,:].std(axis=0)]


  for ax, loss_avg, loss_std, title in zip(axs, results_avg, results_std, titles):
    ax.plot(loss_avg)
    ax.fill_between(x=range(len(loss_avg)), y1=loss_avg+loss_std, y2=loss_avg-loss_std, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    
  plt.show()



def plot_hist_subgroups(n_runs, num_subgroups, dataset, indices, probs, check_bp=False, inv=False):
  bin_centers = np.arange(num_subgroups)
  width = 0.1
  bins = []
  for center in bin_centers:
    bins.append(center-width)
    bins.append(center+width)

  for run in range(n_runs):
    df = pd.DataFrame({'label':dataset.label[indices[run]],
                      'subgroup':dataset.subgroup[indices[run]],
                      'bp':dataset.biased_predictions[indices[run]],
                      'pred':probs[run].argmax(axis=1)})
    df_hist = df.groupby('pred').agg({'subgroup':list})

    fig, axs = plt.subplots(1,len(df_hist),figsize=(21,6))
    if len(df_hist)==1:
      axs = [axs]

    do_check = True
    map_pred_sgs = np.zeros(num_subgroups,dtype=int)
    for i in range(len(df_hist)):
      #sns.histplot(df_hist.iloc[i].subgroup, discrete=True, stat='count', ax=axs[i])
      axs[i].hist(df_hist.iloc[i].subgroup, bins=bins)
      axs[i].set_xlabel('True subgroup label')
      axs[i].set_ylabel('Counts')
      axs[i].set_title('True subgroups in predicted subgroup {}'.format(i))
      u, cnt = np.unique(df_hist.iloc[i].subgroup, return_counts=True)
      cnt_idx = np.argsort(cnt)
      map_pred_sgs[i] = u[cnt_idx[-1]] 
      if len(u) > 1:
        if cnt[cnt_idx[-1]]-cnt[cnt_idx[-2]] < 1000: # for when it fails
          do_check = False
    
    #print(map_pred_sgs)
    if check_bp and do_check:
      if inv:
        df['correct_bp'] = map_pred_sgs[probs[run].argmax(axis=1)]==(~(df.bp.astype(bool))).astype(int)
      else:
        df['correct_bp'] = map_pred_sgs[probs[run].argmax(axis=1)]==df.bp
      df_adv = df.groupby(['label','subgroup']).agg({'correct_bp':'mean'}).rename(columns={'correct_bp':'acc_biased_preds'})
      display(df_adv)
    
  plt.show()

def plot_loss_noadv(epoch_loss, title, runs=list(range(10))):
  epoch_loss_avg = epoch_loss[runs].mean(axis=0)
  epoch_loss_std = epoch_loss[runs].std(axis=0)
  plt.figure(figsize=(9,6))
  plt.plot(epoch_loss_avg)
  plt.fill_between(x=range(epoch_loss.shape[1]), y1=epoch_loss_avg+epoch_loss_std, y2=epoch_loss_avg-epoch_loss_std, alpha=0.6)
  plt.xlabel('Epochs')
  plt.title(title)
  plt.show()
