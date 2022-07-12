from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import numpy as np
import os
import pickle as pkl
import torchvision.transforms as transforms

class CustomDatasetCMNIST(Dataset):
  def __init__(self, train, main_color_prop, class_colors, classes, means=None, stds=None):   
    data = generate_CMNIST(train=train, main_color_prop=main_color_prop, class_colors=class_colors, classes=classes, dataset='mnist')
    self.image = data['images']
    self.label = data['labels']
    self.subgroup = data['colors']
    self.biased_predictions = torch.zeros(self.image.size(0)) # modified later

    if train:
      self.means = self.image.mean(dim=(0,2,3))
      self.stds = self.image.std(dim=(0,2,3))
    else:
      assert (means is not None) and (stds is not None), 'Need to provide means and stds for test set'
      self.means = means
      self.stds = stds

    for i in range(self.image.size(1)):
      self.image[:,i,:,:] = (self.image[:,i,:,:]-self.means[i]) / self.stds[i]

  def __getitem__(self,index):
    image = self.image[index].clone()
    label = self.label[index].clone()
    subgroup = self.subgroup[index].clone()
    biased_pred = self.biased_predictions[index].clone()

    return index, image, label.long(), subgroup.long(), biased_pred.long()


  def __len__(self):
    return self.image.size(0)


class CustomDatasetCFMNIST(Dataset):
  def __init__(self, train, main_color_prop, class_colors, classes, means=None, stds=None):   
    data = generate_CMNIST(train=train, main_color_prop=main_color_prop, class_colors=class_colors, classes=classes, dataset='fashion mnist')
    self.image = data['images']
    self.label = data['labels']
    self.subgroup = data['colors']
    self.biased_predictions = torch.zeros(self.image.size(0)) # modified later

    if train:
      self.means = self.image.mean(dim=(0,2,3))
      self.stds = self.image.std(dim=(0,2,3))
    else:
      assert (means is not None) and (stds is not None), 'Need to provide means and stds for test set'
      self.means = means
      self.stds = stds

    for i in range(self.image.size(1)):
      self.image[:,i,:,:] = (self.image[:,i,:,:]-self.means[i]) / self.stds[i]

  def __getitem__(self,index):
    image = self.image[index].clone()
    label = self.label[index].clone()
    subgroup = self.subgroup[index].clone()
    biased_pred = self.biased_predictions[index].clone()

    return index, image, label.long(), subgroup.long(), biased_pred.long()


  def __len__(self):
    return self.image.size(0)



def inject_ColorBias(images, labels, main_color_prop, class_colors, downsample=True):
  # main_color_prop defines the proportion of images with same color in each class

  n_colors = len(set(class_colors))
  cmap = plt.get_cmap('tab10')
  assert n_colors<=cmap.N, 'No more than {} colors'.format(cmap.N)
  n_classes = len(torch.unique(labels))
  assert n_classes==len(class_colors), 'Provide a color index for each class'
  color_comps = torch.Tensor(cmap.colors) # RGB components for each color
  samples = dict()

  # 2x subsample for computational convenience
  if downsample:
    #images = images.reshape((-1, 28, 28))[:, ::2, ::2] # 
    images = transforms.Resize((14,14))(images)

  colors_all = torch.zeros(len(labels), dtype=torch.long)
  for y,c in enumerate(class_colors):
    p = torch.zeros(n_colors)
    p[torch.arange(n_colors)!=c] = (1-main_color_prop)/(n_colors-1) # probability for other colors
    p[c] = main_color_prop
    colors = torch.multinomial(p, num_samples=(labels==y).sum(), replacement=True)
    colors_all[labels==y] = colors
  

  samples.update(colors=colors_all)
  # Apply the color to the image 
  images = images.float() / 255.
  images = torch.stack([images,images,images],dim=1)
  images =  torch.moveaxis(images, [2,3], [0,1]) # move axis to properly broadcast
  images = images * color_comps[colors_all]
  images =  torch.moveaxis(images, [0,1], [2,3])
      
  samples.update(images=images, labels=labels)
  
  return samples


def generate_CMNIST(train, main_color_prop, class_colors, downsample=True, classes=[4,9], dataset='mnist'):
  arg_s = np.argsort(classes)
  classes = np.array(classes)[arg_s]
  class_colors = np.array(class_colors)[arg_s]
  # get dataset
  if dataset=='mnist':
    mnist = datasets.MNIST('~/datasets/mnist', train=train, download=True)
  elif dataset=='fashion mnist':
    mnist = datasets.FashionMNIST('~/datasets/mnist', train=train, download=True) 

  # keep only selected classes
  sel = mnist.targets==classes[0]
  for c in classes[1:]:
    sel = sel | (mnist.targets==c)
  mnist_data = (mnist.data[sel], mnist.targets[sel])
  # change labels to 0,..,n_classes-1
  for i,c in enumerate(classes):
    mnist_data[1][mnist_data[1]==c] = i

  rng_state = np.random.get_state()
  np.random.shuffle(mnist_data[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_data[1].numpy())

  mnist_biased = inject_ColorBias(mnist_data[0], mnist_data[1], main_color_prop, class_colors, downsample=downsample)

  return mnist_biased


class CustomDatasetCIFAR_S(Dataset):
  def __init__(self, data_split, data_path, normalize=True):   
    self.data_split = data_split
    self.data_path = data_path

    if self.data_split == 'train':
      with open(os.path.join(data_path,'images_train.pkl'), 'rb') as f:
        self.image = pkl.load(f)
      with open(os.path.join(data_path,'labels_train.pkl'), 'rb') as f:
        self.label = pkl.load(f)
      with open(os.path.join(data_path,'domains_train.pkl'), 'rb') as f:
        self.subgroup = pkl.load(f)
    elif self.data_split == 'test':
      with open(os.path.join(data_path,'images_test_gray.pkl'), 'rb') as f:
        image_g = pkl.load(f)
      with open(os.path.join(data_path,'images_test_color.pkl'), 'rb') as f:
        image_c = pkl.load(f)
      with open(os.path.join(data_path,'labels_test.pkl'), 'rb') as f:
        labels = pkl.load(f)
      subgroup_g = np.zeros(image_g.shape[0])
      subgroup_c = np.ones(image_c.shape[0])
      self.image = np.concatenate((image_g,image_c), axis=0)
      self.label = np.concatenate((labels,labels), axis=0)
      self.subgroup = np.concatenate((subgroup_g,subgroup_c), axis=0)
    
    self.biased_predictions = np.zeros(self.image.shape[0]) # modified later

    with open(os.path.join(data_path,'channel_means.pkl'), 'rb') as f:  # cf tests_DBM
      self.means = pkl.load(f)
    with open(os.path.join(data_path,'channel_stds.pkl'), 'rb') as f:
      self.stds = pkl.load(f)

    if normalize:
      self.T = transforms.Compose([ transforms.ToTensor(),
                                    transforms.Normalize(self.means, self.stds),     
                                  ])
    else:
      self.T = transforms.ToTensor()

  def __getitem__(self,index):
    image = self.image[index].copy()
    label = self.label[index].copy()
    subgroup = self.subgroup[index].copy()
    biased_pred = self.biased_predictions[index].copy()

    return index, self.T(image), label.astype(int),  subgroup.astype(int), biased_pred.astype(int)


  def __len__(self):
    return self.image.shape[0]




class CustomDatasetCelebA(Dataset):
  def __init__(self, data_split, data_path):   
    self.data_split = data_split
    self.data_path = data_path

    self.image = torch.load(os.path.join(data_path,data_split+'_images.pt'))
    self.label = torch.load(os.path.join(data_path,data_split+'_labels.pt'))
    self.subgroup = torch.load(os.path.join(data_path,data_split+'_subgroups.pt'))
    self.biased_predictions = torch.zeros(self.label.size(0)) # modified later

    self.T = transforms.Normalize((0.5466, 0.4610, 0.4095),    # cf tests_WILDS
                                  (0.2964, 0.2775, 0.2785))

  def __getitem__(self,index):
    image = self.image[index].clone()
    label = self.label[index].clone()
    subgroup = self.subgroup[index].clone()
    biased_pred = self.biased_predictions[index].clone()

    return index, self.T(image), label.long(),  subgroup.long(), biased_pred.long()


  def __len__(self):
    return self.image.size(0)
