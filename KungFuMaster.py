import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import ale_py
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper


class Network(nn.Module):
    
    def __inti__(self, action_size):
        super(Network, self).__inti__()
        self.conv1 = torch.nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = (3,3), stride = 2) # channels is 4, 3 for RGB and 1 for the grey scale
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
        self.flatten = torch.nn.Flatten() # do the flattening layer after the convolutional layers. No need for batches this time
        self.fc1 = torch.nn.Linear(512, 128) # 512 is the magic number after the flattening. Not really important how this number is calculated but can ask ChatGPT if needed by sharing the code above and it will explain
        # no second hidden layer, only 1 needed, directly connected to the output layer
        self.fc2a = torch.nn.Linear(128, action_size) # this is the output layer of Q Values, also called the Policy. fc2a, a is for "action values"
        self.fc2s = torch.nn.Linear(128, 1) # this is the value of the state used for the critic part V(s) that will be shared between agents. fc2s, s is for "state value"
        
    def forward(self, state): # state will be the input frames, aka the pixels of the image at that frame in rgb and grey scale
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0] # it returns an array of 1 but we want the actual value, so simply access [0]
        return action_values, state_value



class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make('KungFuMasterNoFrameskip-v4', render_mode = 'rgb_array') # The KungFuMaster environment was renamed 'KungFuMasterNoFrameskip-v0'
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.env.get_action_meanings())



# Hyper parameters

learning_rate = 1e-4
discount_factor = 0.99
number_enviroments = 10 # multiple agents running multiple enviromens asynchronous. Training in pararllel will reduce time!

    
class Agent():
    
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # try to use the GPU of NVIDIA. If can't, use CPU
        self.action_size = action_size
        self.brain = Network(action_size).to(self.device) # in this case we don't need a local and a target network, just but the one, called now "brain"
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr = learning_rate)
        
    def act(self, state):
        if state.ndim == 3 # dimension of the state. we work with batches of frames, if this state dimension is 3 it means is a single observation, a single 4 grayscale frame buffer. we need to put it in a batch. (a batch of just itself in it though)... I think??
            state = [state]
        state = torch.tensor(state, dtype = torch.float32, device = self.device) # make it a Pytorch Tensor from numpy array
        
        action_values, _ = self.network(state) # This call the forward() in the network. We don't need the state value yet, so _ for the 2nd param.
        
        policy = F.softmax(action_values, dim = -1) # remember policy are the QValues. Calling the softmax function to get the best action. dim = dimentions to apply softmax to. -1 to do it only on the last dimension (aka the output part of the newral network)... I think??
        
        return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()]) # policy is a tensor. detach from the ccomputational grapch (?). move the tensor back to CPU. Convert that tensor into a numpyarray. then choice selects the action with most value from the policy. Do this for each state of the batch
    
