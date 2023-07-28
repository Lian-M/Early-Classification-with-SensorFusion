from torch import nn
import torch
from torch.distributions import Bernoulli

class BaselineNetwork(nn.Module):
    """
    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, h_t):
        b_t = self.fc(h_t.detach()) # b_t = torch.sigmoid(self.fc(h_t.detach()))
        return b_t

class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward

    def forward(self, h, eps=0.):
        """Read in hidden state, predict one halting probability per class"""
        # Predict one probability per class
        probs = torch.sigmoid(self.fc(h))  # the original input x should be replaced by h 此句的h原为x，可能写错了？，sigmoid激活函数的输出应该就是此时网络根据隐藏状态ht与分类网络输出y_hat通过全连接层生成的对各个类别的分类信念

        # Balance between explore/exploit by randomly picking some actions
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05])  # Explore/exploit (can't be 0), 如果self._epsilon=0则每一类的预测概率在probs中都是确定的，就无法引入随机性进行explore，
        # Parameterize bernoulli distribution with prediced probabilities
        m = Bernoulli(probs=probs)

        # Sample an action and compute the log probability of that action being
        # picked (for use during optimization)
        action = m.sample() # sample an action 利用网络对各个类别的分类信念进行相应的操作
        log_pi = m.log_prob(action) # compute log probability of sampled action

        # We also return the negative log probability of the probabilities themselves 这个可以视为uncertainty，因为最小化这个值就是最大化halting的似然，也就是最大化Confidence
        # if we minimize this, it will maximize the likelihood of halting!
        return action, log_pi, -torch.log(probs)
