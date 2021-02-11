import torch as tc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

# RNN class
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        # define the network modules here
        # e.g. self.layer = nn.Linear(6, 5)        
        self.hidden_size = hidden_size
            
        self.w_xz = nn.Parameter( tc.Tensor(input_size, hidden_size) )
        self.w_zz = nn.Parameter( tc.Tensor(hidden_size, hidden_size) )
        self.w_zx = nn.Parameter( tc.Tensor(hidden_size, input_size) )
        
        self.b_z = nn.Parameter( tc.Tensor(hidden_size) )
        self.b_x = nn.Parameter( tc.Tensor(input_size) )
        
        self.init_weights()
        
        
    def init_weights(self):
        tc.manual_seed(2) # for reproducibility
        
        for name, p in self.state_dict().items():
            nn.init.uniform_(p.data)
                

    def forward(self, inp, hidden):
        # instantiate modules here
        # e.g. output = self.layer(inp)

        hidden = tc.tanh( tc.mm(inp,self.w_xz) + tc.mm(hidden,self.w_zz) + self.b_z )
        output = tc.mm(hidden,self.w_zx) + self.b_x

        return output, hidden
    
    
    def get_prediction(self, inp, T):
        hidden = tc.zeros((1, self.hidden_size))
        predictions = []

        for i in range(T):  # predict for longer time than the training data
            prediction, hidden = self.forward(inp, hidden)
            inp = prediction
            predictions.append(prediction.data.numpy().ravel()[0])

        return predictions
    
    
    def train(self, x, y, lr, epochs, lmbd):
        if lmbd == 0:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay = lmbd)
        
        losses = []
        
        for i in range(epochs):
            hidden = tc.zeros((1, self.hidden_size))
            for j in range(x.size(0)):
                optimizer.zero_grad()
                input_ = x[j:(j+1)]
                target = y[j:(j+1)]
                (prediction, hidden) = self.forward(input_, hidden)
                loss = (prediction - target).pow(2).sum()/2

                loss.backward(retain_graph=True)  # retain, because of BPTT (next lecture)
                optimizer.step()
                #losses.append(loss)
            losses.append(loss)
            
        return losses

    
# define rnn function to be called
def rnn(hidden_size = 10, lr=0.01, epochs = 400, lmbd = 0):
    
    # load data
    data = tc.load('noisy_sinus.pt')
    x = tc.FloatTensor(data[:-1])
    y = tc.FloatTensor(data[1:])
    
    # create RNN model
    model = RNN(input_size=1, hidden_size=hidden_size, output_size=1)

    # train RNN model
    losses = model.train(x, y, lr, epochs, lmbd)

    
    # plots
    fig1 = plt.subplots(figsize=[16,5])
    
    # print title
    s_lr = ('%f' % lr).rstrip('0').rstrip('.')
    if lmbd == 0:
        plt.suptitle('RNN with h.size=%i, rate=%s' %(hidden_size, s_lr), fontsize = 20)
    else:
        s_lmbd = ('%f' % lmbd).rstrip('0').rstrip('.')
        plt.suptitle(r'RNN with h.size=%i, rate=%s, $\lambda=%s$' %(hidden_size, s_lr, s_lmbd), fontsize = 20)
    plt.subplots_adjust(hspace = 0.3)

    # plot MSE
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.title('Loss', fontsize = 16)
    plt.xlabel('epoch', fontsize = 12)
    plt.ylabel('MSE', fontsize = 12)
    plt.grid()
    
    # plot predictions over true data
    plt.subplot(1,2,2)
    predictions = model.get_prediction(inp=x[0:1],T=6*x.size(0))
    plt.plot(data[1:], label='true data')
    plt.plot(predictions, label='generated data')
    plt.xlabel('timestep t', fontsize = 12)
    plt.ylabel(r'$x_t$', fontsize = 12)
    plt.title('Data and prediction', fontsize = 16)
    plt.grid()
    plt.legend()
    
    plt.show(fig1)

