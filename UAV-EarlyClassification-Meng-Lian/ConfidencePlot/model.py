import torch
from torch import nn
from modules import *
import numpy as np
#from loss import * 

class RHC(nn.Module):
    def __init__(self, ninp, nhid, nclasses, nlayers, nepochs=200, lam=0.0):     # 原程序中此处nepochs=0
        """
        Recurrent Halting Chain, as proposed in "Recurrent Halting Chain for
        Early Multi-label Classification", published at KDD 2020.

        The main choice to be made is how much emphasis to place on earliness.
        This is controlled by self.LAMBDA. When lambda=0, this means "halt only
        when it helps". As lambda grows, more emphasis is put on halting early.
        We use a log-scale search from 0 to 0.1 in 11 steps in the paper, which
        covers the whole frontier. 

        Usage of the forward method
        ---------------------------

        Parameters
        ----------
        X : torch.tensor of shape (timesteps x instances x variables) T, B, V
            This is the time series input. As implemented, this method requires
            equal-length time series.
        test : boolean (前向传播函数forward中，用于调整网络运行模式，True means test mode while False means train mode)
            This tells the model whether or not it's testing time. This lets us
            do different things during training and testing time. For instance,
            in this code, we schedule the explore/exploit trade-off during
            training.

        Returns
        -------
        y_hat : torch.tensor of shape (instances x total number of classes)
            These are the class probabilities predicted by RHC.
        mean_halting_point: scalar
            On average how much each class was halted over the whole batch.
        """
        super(RHC, self).__init__()

        # --- Hyperparameters ---
        self.LAMBDA = lam
        self.nepochs = nepochs
        self.nlayers = nlayers
        self.nhid = nhid
        self._N_CLASSES = nclasses                          # this scentence is missed by the original code 原程序中缺少这一句定义
        self._exponentials = self.exponentialDecay(nepochs) # use an ϵ-greedy approach to choose between the predicted halting decision and a randomly-selected action, as shown in Equation
                                                            # 14 where ϵ is 1 at the beginning of training and decreases to 0 epponentially throughout training.

        # --- Sub-networks ---
        self.Controller = Controller(nclasses + nhid + 1, nclasses) # Halting Policy Network (where uses reinforcement learning and ϵ-greedy approach)
        self.BaselineNetwork = BaselineNetwork(nhid, nclasses)      # Discriminator Network
        self.RNN = torch.nn.LSTM(ninp + nclasses, nhid)             # Transition Model
        self.out = torch.nn.Linear(nhid, nclasses)                  # Output Layer

    def exponentialDecay(self, N):
        tau = 1
        tmax = 7
        t = np.linspace(0, tmax, N)
        y = torch.tensor(np.exp(-t/tau), dtype=torch.float)
        return y

    def forward(self, X, epoch, test=False):
        if test:
            # No random decisions during testing!
            self.Controller._epsilon = 0.0
        else:
            self.Controller._epsilon = self._exponentials[epoch]  # Explore/exploit
        T, B, V = X.shape # T: timesteps, B: number of time series, V: dimension of time series(number of variables).# where X is a tensor of time series of shape (timesteps, instances, variables)
        baselines = []    # Predicted baselines
        actions = []      # Which classes to halt at each step
        log_pi = []       # Log probability of chosen actions
        halt_probs = []
        halt_points = -torch.ones((B, self._N_CLASSES))
        y_bar = torch.zeros((1, B, self._N_CLASSES), requires_grad=False)      # Indicator vector
        hidden = self.initHidden(X.shape[1])                                   # Hidden state
        predictions = torch.zeros((1, B, self._N_CLASSES), requires_grad=True) # Record predicted values
        all_preds = []

        # --- for each timestep, select a set of actions ---
        for t in range(T):
            RNN_in = torch.cat((X[t].unsqueeze(0), y_bar.detach()), dim=2)
            state, hidden = self.RNN(RNN_in, hidden)
            y_hat = torch.sigmoid(self.out(state)) # y_hat : torch.tensor of shape (instances x total number of classes), real-time soft confidence实时软概率
            all_preds.append(y_hat)
            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1, 1, 1).repeat(1, B, 1)  # collect timestep,  build the matrix of time step for the nextconcat operation 为了下一步的拼接（与state和y_hat同时作为控制器输入）做准备。
            c_in = torch.cat((state, y_hat, time), dim=2).detach()
            a_t, p_t, w_t = self.Controller(c_in)  # Compute halting-probability and sample an action，a_t is the sampled action '0' or '1', p_t is log probability of the sampled action
            predictions = torch.where((a_t == 1) & (predictions == 0), y_hat, predictions) # record predicted probabilities if one class is halted at a time step在某一时间步，当某一类halt了(成功预测)记录此类当前的软概率；如果某一类还没有halt（预测成功）那么其在predictions中的值保持为0，直到某个时间步其halt了才记录此类当前软概率，并在其后的时间步上保持不变。所以说predictions记录的是某一类分类成功时的概率，并不是实时概率。
            y_bar = torch.where((a_t == 1) & (y_bar == 0), torch.ones_like(y_bar), y_bar)
            halt_points = torch.where((halt_points == -1) & (a_t == 1), time, halt_points) # record the time step if one class is halted
            #print('The classification result at time step {} is: {}'.format(t+1, 1+halt_points)) # value "0" means this class has not been classified clearly，value "5" means this class has been clearly classified at 5th time step.
            b_t = self.BaselineNetwork(state) # soft confidence at time step t
            actions.append(a_t.squeeze())   # Which classes to halt at each step
            baselines.append(b_t)           # Predicted baselines
            log_pi.append(p_t)              # Log probability of chosen actions
            halt_probs.append(w_t)          # Probability of halt action
            #print(np.shape(y_hat))
            # Soft confidence compute
            # Calculate the softmax over the outputs to obtain a probability distribution
            softmax = nn.Softmax(dim=1)
            y_hatprobs = softmax(y_hat[0,...,...]) # np.shape(y_hatprobs) = [49, 6] (batch, nclasses)
            # Calculate the entropy of the distribution to estimate soft confidence
            entropy = -torch.sum(y_hatprobs * torch.log(y_hatprobs), dim=1) #[49]
            confidence = entropy.mean()
            #confidence = -torch.mean(y_hatprobs * torch.log(y_hatprobs), dim=0) #[49]

            #print('The soft confidence at time step {} is: {}'.format(t+1, y_hat[0,0,...]*torch.sigmoid(w_t[0,0,...])))
            #print('The soft confidence discriminator produced at time step {} is: {}'.format(t+1, y_hat[0, 0,...]))
            #print('The soft confidence discriminator produced at time step {} is: {}'.format(t+1, confidence))
            if (halt_points == -1).sum() == 0:  # If no negative values, it means all classes has been halted (successful classify), then stop running the next time step operation
                break
                
        #self.e = entropy_mean
        self.seq_preds = torch.stack(all_preds).squeeze() # predicted classification probabilities after running all time steps
        self.y_hat = torch.where(predictions == 0.0, y_hat, predictions).squeeze()  # If it never stopped to predict, use final prediction
        halt_points = torch.where(halt_points == -1, time, halt_points).squeeze()   # If it never stopped to predict, assume the last timestep was when it halted (waiting until the end is the same as halting at the end)
        self.locations = np.array(halt_points + 1)
        self.baselines = torch.stack(baselines)  # .transpose(1, 0)#.view(1, -1)
        self.log_pi = torch.stack(log_pi).squeeze(1)  # .transpose(1, 0)
        self.halt_probs = torch.stack(halt_probs)  # .transpose(1, 0)
        self.actions = torch.stack(actions)  # .transpose(1, 0)

        # --- Compute mask for where actions are updated ---
        self.grad_mask = torch.zeros_like(self.actions) # used to compute loss
        for i in range(B):  # Batch
            for k in range(self._N_CLASSES):  # Classes
                self.grad_mask[:(1 + halt_points[i, k]).long(), i, k] = 1
        mean_halting_point = (1+halt_points.detach()).mean()/(T+1) # achieve mean value over two (batch and class) dimension, mean_halting_step = mean_halting_point(0~1 scalar) * (T+1)
        return self.y_hat.squeeze(), mean_halting_point #, halt_points

    def initHidden(self, bsz):
        """Initialize hidden states"""
        h = (torch.zeros(self.nlayers,
                         bsz,
                         self.nhid),
             torch.zeros(self.nlayers,
                         bsz,
                         self.nhid))
        return h

    def computeLoss(self, y_hat, y):
        """Compute the loss function"""
        # --- compute reward ---
        self.r = (2*(y_hat.float().round() == y.float()).float()-1).detach()
        self.R = self.r * self.grad_mask
        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines.squeeze(1)
        self.adjusted_reward = self.R - b.detach()
        
        # --- compute losses ---
        MSE = torch.nn.MSELoss()
        BCE = torch.nn.BCELoss()
        #PrIOU = MSE(y_hat, y)
        #confidence = (y_hat.detach()* torch.sigmoid(self.halt_probs.detach())).mean() # defined as confidence
        
        self.loss_b = MSE(b, self.R) # Baseline should approximate mean reward
        self.loss_r = (-self.log_pi*self.adjusted_reward).sum(0).mean()
        self.loss_c = BCE(y_hat, y.float())
        self.wait_penalty = self.halt_probs.squeeze(1).sum(0).sum(1).mean() # time, classes, batch mean
        self.LAMBDA = torch.tensor([self.LAMBDA], dtype=torch.float, requires_grad=False)
        loss = self.loss_r + self.loss_b + self.LAMBDA*(self.wait_penalty) + self.loss_c  # use Hyperparameter lam to control the loss induced by wait (not halt) 
        return loss
