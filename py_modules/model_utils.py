import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import confusion_matrix, homogeneity_score, completeness_score
import itertools
import pandas as pd
from models import MLP, CNN, CNN2, CNN3
import copy
import scipy

def train_reference(model, data_loader, optimizer, loss_fn, n_epochs, device, verbose=True):
  model.train()
  loss_epoch = np.zeros(n_epochs)
  for epoch in range(n_epochs):
    for _, images, labels, _, _ in data_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      out, _ = model(images)
      loss = loss_fn(out, labels)
      # update model
      loss.backward()
      optimizer.step()
      # monitor loss
      loss_epoch[epoch] += loss.detach().cpu().numpy()

    if verbose:
      if (not epoch%10) or (epoch==n_epochs-1):
        print('Epoch {}: loss={:.5f}'.format(epoch, loss_epoch[epoch]/ len(data_loader)))
  loss_epoch = loss_epoch / len(data_loader)

  return model, loss_epoch

def test_reference(model, data_loader, device): 
  model.eval()
  softmax = nn.Softmax(dim=-1)
  correct = 0
  nsamples = 0
  preds_all = []
  labels_all = []
  subgroup_all = []
  for _, images, labels, sg, _ in data_loader: 
    images, labels = images.to(device), labels.to(device)
    nsamples += labels.size(0)
    out, _ = model(images)
    out = softmax(out)
    pred = out.argmax(dim=-1)
    correct += (pred==labels).sum()
    preds_all = preds_all + pred.cpu().tolist()
    labels_all = labels_all + labels.cpu().tolist()
    subgroup_all = subgroup_all + sg.tolist()
  accuracy = correct/nsamples
  print('Average accuracy: {:.5f}'.format(accuracy))

  df = pd.DataFrame({'prediction': preds_all, 'label':labels_all, 'subgroup':subgroup_all})
  df['correct'] = df.prediction==df.label
  n_subgroups = len(np.unique(subgroup_all))
  confs = []
  for subg in range(n_subgroups):
    confs.append(confusion_matrix(df.loc[df.subgroup==subg, 'label'], df.loc[df.subgroup==subg, 'prediction']))

  return df.groupby(['label', 'subgroup']).agg({'correct':'mean'}).rename(columns={'correct':'accuracy'}), confs
  

class DemographicParityGap(nn.Module):
  def __init__(self, agg=None, print_gaps=False):
    super(DemographicParityGap, self).__init__()
    self.agg=agg
    self.print_gaps = print_gaps

  def forward(self, output, biased_predictions, labels, num_classes, num_subgroups, device):
    # labels not used here, added for consistency with eq. odds
    bp_one_hot = nn.functional.one_hot(biased_predictions,num_classes=num_classes) # specify num_classes in case some are absent in bp 
    classes_one_hot = nn.functional.one_hot(torch.arange(0,num_classes), num_classes=num_classes) 
    classes_one_hot = classes_one_hot.to(device)
    n_combs = int(scipy.special.binom(num_subgroups, 2))

    dpgs = torch.zeros((num_classes,n_combs))
    for i,class_one_hot in enumerate(classes_one_hot):
      j=0
      for sg0,sg1 in itertools.combinations(np.arange(num_subgroups), r=2):
        demP0 = ((class_one_hot*bp_one_hot).sum(dim=1)*output[:,sg0]).sum()/(output[:,sg0].sum())
        demP1 = ((class_one_hot*bp_one_hot).sum(dim=1)*output[:,sg1]).sum()/(output[:,sg1].sum())
        dpgs[i,j] =  (demP0 - demP1)**2
        j += 1
    
    if self.agg is None:
      loss = dpgs.sum() / (num_classes*n_combs)
    elif self.agg == 'min_sg':
      min, _ =  dpgs.min(dim=1)
      loss = min.sum()/num_classes
    elif self.agg == 'min':
      loss = dpgs.min()

    if self.print_gaps:
      print(dpgs)
    
    return  -loss


class EqualizedOddsGap(nn.Module):
  def __init__(self, agg=None, print_gaps=False):
    super(EqualizedOddsGap, self).__init__()
    self.agg=agg
    self.print_gaps = print_gaps

  def forward(self, output, biased_predictions, labels, num_classes, num_subgroups, device):
    bp_one_hot = nn.functional.one_hot(biased_predictions,num_classes=num_classes) # specify num_classes in case some are absent in bp 
    labels_one_hot = nn.functional.one_hot(labels,num_classes=num_classes)
    classes_one_hot = nn.functional.one_hot(torch.arange(0,num_classes), num_classes=num_classes) 
    classes_one_hot = classes_one_hot.to(device)
    n_combs = int(scipy.special.binom(num_subgroups, 2))

    gaps = torch.zeros((num_classes,n_combs))
    for i,class_one_hot in enumerate(classes_one_hot):
      other_classes = (~(class_one_hot.bool())).float()
      j=0
      for sg0,sg1 in itertools.combinations(np.arange(num_subgroups), r=2):
        demP0 = ((other_classes*bp_one_hot).sum(dim=1)*(class_one_hot*labels_one_hot).sum(dim=1)*output[:,sg0]).sum() / (((class_one_hot*labels_one_hot).sum(dim=1)*output[:,sg0]).sum())
        demP1 = ((other_classes*bp_one_hot).sum(dim=1)*(class_one_hot*labels_one_hot).sum(dim=1)*output[:,sg1]).sum() / (((class_one_hot*labels_one_hot).sum(dim=1)*output[:,sg1]).sum())
        gaps[i,j] =  (demP0 - demP1)**2
        j += 1
    
    if self.agg is None:
      loss = gaps.sum() / (num_classes*n_combs)
    elif self.agg == 'min_sg':
      min, _ = gaps.min(dim=1)
      loss = min.sum()/num_classes
    elif self.agg == 'min':
      loss = gaps.min()

    if self.print_gaps:
      print(gaps)
    
    return  -loss

class GradReverse(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.neg() * 1 # probably can be set to 1 ?

def grad_reverse(x):
  return GradReverse.apply(x)


def train_subgroup(model, optimizer, loss_fn, adversary, optimizer_adv, loss_fn_adv, 
                   data_loader, n_epochs, device, _lambda=0.001, verbose=True, 
                   test_probs_bp=False, num_classes=2, num_subgroups=2, anomaly_detection=False):
  model.train()
  adversary.train()
  epoch_loss = torch.zeros(n_epochs)
  epoch_mi = torch.zeros(n_epochs)
  epoch_loss_adv = torch.zeros(n_epochs)
  softmax = nn.Softmax(dim=-1).to(device)
  torch.autograd.set_detect_anomaly(anomaly_detection)

  for epoch in range(n_epochs):
    for _, images, labels, subgroups, biased_preds in data_loader: 
      images, labels, biased_preds = images.to(device), labels.to(device), biased_preds.to(device)
      optimizer.zero_grad()
      optimizer_adv.zero_grad()
      # compute outputs and losses
      out, features = model(images)
      out = softmax(out) 
      out_adv, _ = adversary(features) 
      out_adv =  softmax(out_adv)
      if out.isnan().any():
        print('NANS in forward learner! epoch {}'.format(epoch))
      if out_adv.isnan().any():
        print('NANS in forward adv! epoch {}'.format(epoch))
      if (out_adv<=-(1e-20)).any():
        print('Future NANS in mi reg! epoch {}'.format(epoch))
      #loss_dpg = loss_fn(out, biased_preds) # old one with bin bin dpg 
      loss_dpg = loss_fn(out, biased_preds, labels, num_classes, num_subgroups, device)
      mi_reg = torch.mean(torch.sum(out_adv*torch.log(out_adv+1e-20), 1))
      loss = loss_dpg + _lambda*mi_reg
      # update learner
      loss.backward()
      optimizer.step()

      optimizer.zero_grad()
      optimizer_adv.zero_grad()
      # compute outputs and losses
      _, features = model(images)
      features = grad_reverse(features)
      out_adv, _ = adversary(features)
      if out_adv.isnan().any():
        print('NANS in forward adv2! epoch {}'.format(epoch))
      if test_probs_bp:
        loss_adv = loss_fn_adv(out_adv, (biased_preds>0.5).long())
      else:
        loss_adv = loss_fn_adv(out_adv, biased_preds)
      # update learner and adversary
      loss_adv.backward()
      optimizer.step() 
      optimizer_adv.step()

      # monitor losses
      epoch_loss[epoch] += -loss_dpg.cpu()
      epoch_mi[epoch] += mi_reg.cpu()
      epoch_loss_adv[epoch] += loss_adv.cpu()

      check_nans = loss_dpg.isnan() | mi_reg.isnan() | loss_adv.isnan()
      if check_nans:
        print('NANS! Epoch {}: dpg={} mi_reg={} loss_adv={}'.format(epoch, -loss_dpg, mi_reg, loss_adv))
        print('Previous epoch {}: dpg={} mi_reg={} loss_adv={}'.format(epoch-1, epoch_loss[epoch-1], epoch_mi[epoch-1], epoch_loss_adv[epoch-1]))
        epoch_loss /= len(data_loader)
        epoch_mi /= len(data_loader)
        epoch_loss_adv /= len(data_loader)
        return model, adversary, [epoch_loss.detach().numpy(), epoch_mi.detach().numpy(), epoch_loss_adv.detach().numpy()]

    if verbose and ((not epoch%5) or (epoch==n_epochs-1)):
      print('Epoch {}: dpg={} mi_reg={} loss_adv={}'.format(epoch, epoch_loss[epoch]/len(data_loader), 
                                                            epoch_mi[epoch]/len(data_loader), epoch_loss_adv[epoch]/len(data_loader)))

  epoch_loss /= len(data_loader)
  epoch_mi /= len(data_loader)
  epoch_loss_adv /= len(data_loader)

  return model, adversary, [epoch_loss.detach().numpy(), epoch_mi.detach().numpy(), epoch_loss_adv.detach().numpy()]


def train_subgroup_noadv(model, optimizer, loss_fn, data_loader, n_epochs, device, num_classes, num_subgroups, verbose=False):
  model.train()
  epoch_loss = torch.zeros(n_epochs)
  softmax = nn.Softmax(dim=-1)

  for epoch in range(n_epochs):
    for _, images, labels, subgroups, biased_preds in data_loader: 
      images, labels, biased_preds = images.to(device), labels.to(device), biased_preds.to(device)
      optimizer.zero_grad()
      # compute outputs and losses
      out, features = model(images)
      out = softmax(out) 
      loss_dpg = loss_fn(out, biased_preds, labels, num_classes, num_subgroups, device)
      # update learner
      loss_dpg.backward()
      optimizer.step()
      # monitor losses
      epoch_loss[epoch] += -loss_dpg.cpu()

      if loss_dpg.isnan():
        print('NANS! Epoch {}: dpg={}'.format(epoch, -loss_dpg.cpu()))
        epoch_loss /= len(data_loader)
        return model, epoch_loss.detach().numpy()

    if verbose:
      if (not epoch%5) or (epoch==n_epochs-1):
        print('Epoch {}: dpg={} '.format(epoch, epoch_loss[epoch]/len(data_loader)))

  epoch_loss /= len(data_loader)

  return model, epoch_loss.detach().numpy()


def test_subgroup(model, data_loader, device, adversary=None):
  model.eval()
  if adversary is not None:
    adversary.eval()
    correct_adv = 0
  softmax = nn.Softmax(dim=-1)
  probs_all =[] #aux
  idx_all = []
  nsamples = 0
  
  for idx, images, labels, sg, bp in data_loader: 
    nsamples += labels.size(0)
    images, sg, bp = images.to(device), sg.to(device), bp.to(device)
    out, feats = model(images)
    out = softmax(out)
    probs_all = probs_all + out.tolist()
    idx_all = idx_all + idx.tolist()
    
    if adversary is not None:
      out_adv, _ = adversary(feats)
      out_adv = softmax(out_adv)
      pred_adv = out_adv.argmax(dim=-1)
      correct_adv += (pred_adv==bp).sum()

  
  if adversary is not None:
    return np.asarray(probs_all), np.asarray(idx_all), (correct_adv/nsamples).item()
  else:
    return np.asarray(probs_all), np.asarray(idx_all)



def n_runs_subgroupNoAdv(n_runs, n_epochs, num_classes, num_subgroups, dataset_tr, dataset_te,
                         lr, loss_fn, train_loader, test_loader, device, model, verbose=False, 
                         same_weight_init=False):

  nsamples_tr = dataset_tr.__len__()
  nsamples_te = dataset_te.__len__()

  epoch_loss = np.zeros((n_runs,n_epochs))
  scores_tr = np.zeros((n_runs,2))
  scores_te = np.zeros((n_runs,2))
  probs_tr = np.zeros((n_runs, nsamples_tr, num_subgroups))
  idx_tr = np.zeros((n_runs, nsamples_tr), dtype=int)
  probs_te = np.zeros((n_runs, nsamples_te, num_subgroups))
  idx_te = np.zeros((n_runs, nsamples_te), dtype=int)

  for run in range(n_runs):
    print('--------Run %d--------'%run)
    if not same_weight_init:
      if model.name=='mlp':
        subgroup_clf = MLP(model.args[0], model.args[1], model.args[2], model.args[3]).to(device)
      elif model.name=='cnn':
        subgroup_clf = CNN(model.args[0]).to(device)
      elif model.name=='cnn3':
        subgroup_clf = CNN3(model.args[0]).to(device)
    else:
      subgroup_clf = copy.deepcopy(model).to(device)

    optimizer = optim.Adam(subgroup_clf.parameters(), lr=lr) #
    #optimizer = optim.SGD(subgroup_clf.parameters(), lr=0.01)

    subgroup_clf, epoch_loss[run] = train_subgroup_noadv(subgroup_clf, optimizer, loss_fn, train_loader, n_epochs, 
                                                         device, num_classes=num_classes, num_subgroups=num_subgroups, verbose=verbose)
    probs_tr[run], idx_tr[run] = test_subgroup(subgroup_clf, train_loader, device)
    pred_tr = probs_tr[run].argmax(axis=1) 
    scores_tr[run,0] = homogeneity_score(dataset_tr.subgroup[idx_tr[run]], pred_tr)
    scores_tr[run,1] = completeness_score(dataset_tr.subgroup[idx_tr[run]], pred_tr)
    probs_te[run], idx_te[run], = test_subgroup(subgroup_clf, test_loader, device)
    pred_te = probs_te[run].argmax(axis=1)
    scores_te[run,0] = homogeneity_score(dataset_te.subgroup[idx_te[run]], pred_te)
    scores_te[run,1] = completeness_score(dataset_te.subgroup[idx_te[run]], pred_te)
    print('Train homogeneity score: {:.3f} ; Train completeness score: {:.3f}'.format(scores_tr[run,0],scores_tr[run,1]))
    print('Test homogeneity score: {:.3f} ; Test completeness score: {:.3f}'.format(scores_te[run,0],scores_te[run,1]))

  return {'epoch_loss':epoch_loss, 'scores_tr':scores_tr, 'scores_te':scores_te,
          'probs_tr':probs_tr, 'idx_tr':idx_tr, 'probs_te':probs_te, 'idx_te':idx_te}   


def n_runs_subgroupAdvTrain_v2(n_runs, n_epochs, dataset_tr, dataset_te, device, num_classes, num_subgroups, train_loader, test_loader,
                               model_learner, model_adv, loss_fn_learner=DemographicParityGap(), loss_fn_adv=nn.CrossEntropyLoss(), 
                               lr=0.01, lr_adv=0.001, mi_weight=0.1, verbose=False, same_weight_init=False, l2_reg=0, anomaly_detection=False):
    
  nsamples_tr = dataset_tr.__len__()
  nsamples_te = dataset_te.__len__()
  epoch_loss = np.zeros((n_runs,n_epochs))
  epoch_mi = np.zeros((n_runs,n_epochs))
  epoch_loss_adv = np.zeros((n_runs,n_epochs))
  scores_tr = np.zeros((n_runs,2))
  scores_te = np.zeros((n_runs,2))
  accuracies_tr_adv = np.zeros(n_runs)

  probs_tr = np.zeros((n_runs, nsamples_tr, num_subgroups))
  idx_tr = np.zeros((n_runs, nsamples_tr), dtype=int)
  probs_te = np.zeros((n_runs, nsamples_te, num_subgroups))
  idx_te = np.zeros((n_runs, nsamples_te), dtype=int)


  for run in range(n_runs):
    print('--------Run %d--------' %run)

    if not same_weight_init:
      if model_learner.name=='mlp':
        subgroup_clf = MLP(model_learner.args[0], model_learner.args[1], model_learner.args[2], model_learner.args[3]).to(device)
      elif model_learner.name=='cnn':
        subgroup_clf = CNN(model_learner.args[0]).to(device)
      elif model_learner.name=='cnn3':
        subgroup_clf = CNN3(model_learner.args[0]).to(device)
      adversary = MLP(model_adv.args[0], model_adv.args[1], model_adv.args[2], model_adv.args[3]).to(device) # adversary is always mlp
    else:
      subgroup_clf = copy.deepcopy(model_learner).to(device)
      adversary =  copy.deepcopy(model_adv).to(device)

    optimizer = optim.SGD(subgroup_clf.parameters(), lr=lr, weight_decay=l2_reg)
    optimizer_adv = optim.SGD(adversary.parameters(), lr=lr_adv) 
    subgroup_clf, adversary, results = train_subgroup(subgroup_clf, optimizer, loss_fn_learner, adversary, optimizer_adv, loss_fn_adv,
                                                      train_loader, n_epochs, device, _lambda=mi_weight, verbose=verbose,
                                                      num_classes=num_classes, num_subgroups=num_subgroups, anomaly_detection=anomaly_detection) #gradreverse  1.
    epoch_loss[run] = results[0]
    epoch_mi[run] = results[1]
    epoch_loss_adv[run] = results[2]

    # run scores
    probs_tr[run], idx_tr[run], accuracies_tr_adv[run] = test_subgroup(subgroup_clf, train_loader, device, adversary)
    pred_tr = probs_tr[run].argmax(axis=1) 
    scores_tr[run,0] = homogeneity_score(dataset_tr.subgroup[idx_tr[run]], pred_tr)
    scores_tr[run,1] = completeness_score(dataset_tr.subgroup[idx_tr[run]], pred_tr)
    probs_te[run], idx_te[run] = test_subgroup(subgroup_clf, test_loader, device)
    pred_te = probs_te[run].argmax(axis=1)
    scores_te[run,0] = homogeneity_score(dataset_te.subgroup[idx_te[run]], pred_te)
    scores_te[run,1] = completeness_score(dataset_te.subgroup[idx_te[run]], pred_te)
    print('Train homogeneity score: {:.3f} ; Train completeness score: {:.3f}'.format(scores_tr[run,0],scores_tr[run,1]))
    print('Test homogeneity score: {:.3f} ; Test completeness score: {:.3f}'.format(scores_te[run,0],scores_te[run,1]))
    print('Adversary train accuracy (target=biased predictions): {:.5f}'.format(accuracies_tr_adv[run]))

  return {'epoch_loss':epoch_loss, 'epoch_mi':epoch_mi, 'epoch_loss_adv':epoch_loss_adv, 
          'scores_tr':scores_tr, 'scores_te':scores_te, 'accuracies_tr_adv':accuracies_tr_adv,
          'probs_tr':probs_tr, 'idx_tr':idx_tr, 'probs_te':probs_te, 'idx_te':idx_te}
    
