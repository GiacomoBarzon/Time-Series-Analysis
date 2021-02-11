import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, dim_x, dim_h, dim_z):
        super().__init__()
        self.linear = nn.Linear(dim_x, dim_h)
        self.mu = nn.Linear(dim_h, dim_z)
        self.logvar = nn.Linear(dim_h, dim_z)

    def forward(self, x):
        h = F.relu(self.linear(x))
        z_mu = self.mu(h)
        z_logvar = self.logvar(h)
        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, dim_z, dim_h, dim_x):
        super().__init__()
        self.linear1 = nn.Linear(dim_z, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_x)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        x = tc.sigmoid(self.linear2(h))
        return x


class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z_mu, z_logvar = self.enc(x)
        z_sample = reparametrize(z_mu, z_logvar)
        x_sample = self.dec(z_sample)
        return x_sample, z_mu, z_logvar


def reparametrize(z_mu, z_logvar):
    # NOTE: the 'reparametrization trick' will be treated next week
    # essentially it is sampling from a Gaussian distribution,
    # but still being differentiable w.r.t. the parameters mu, Sigma
    std = tc.exp(z_logvar)
    eps = tc.randn_like(std)
    x_sample = eps.mul(std).add_(z_mu)
    return x_sample


def loss_function(x, x_sample, z_mu, z_logvar):
    rec_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')
    kl_loss = 0.5 * tc.sum(z_mu**2 + tc.exp(z_logvar) - tc.ones(dim_z) - z_logvar)
    return rec_loss + kl_loss


def train():
    model.train()
    train_loss = 0
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        x = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        x_sample, z_mu, z_logvar = model(x)
        loss = loss_function(x, x_sample, z_mu, z_logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss


def test():
    model.eval()
    test_loss = 0
    with tc.no_grad():  # no need to track the gradients here
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(-1, 28 * 28)
            z_sample, z_mu, z_var = model(x)
            loss = loss_function(x, z_sample, z_mu, z_var)
            test_loss += loss.item()
    return test_loss


def generate():
    sample = tc.randn(generated, dim_z)
    return model.dec.forward(sample)


if __name__ == '__main__':
    batch_size = 64         # number of data points in each batch
    n_epochs = 1          # times to run the model on complete data
    dim_x = 28 * 28         # size of each input
    dim_h = 256             # hidden dimension
    dim_z = 50              # latent vector dimension
    lr = 1e-3               # learning rate
    generated = 64          # number of generated images

    # import dataset
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    transforms = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # initialize VAE model
    encoder = Encoder(dim_x, dim_h, dim_z)
    decoder = Decoder(dim_z, dim_h, dim_x)
    model = VAE(encoder, decoder).to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # network training
    ELBO = []
    
    for epoch in range(n_epochs):
        train_loss = train()
        test_loss = test()
        train_loss /= len(train_set)
        test_loss /= len(test_set)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
        ELBO.append(-train_loss)

    # plot ELBO
    fig1 = plt.subplots(figsize=[16,8])

    plt.plot(list(range(1,n_epochs+1)), ELBO, 'o')
    plt.xlabel('epochs', fontsize = 18)
    plt.ylabel('ELBO', fontsize = 18)
    plt.grid()
    plt.title('ELBO vs. epochs', fontsize=20)
    plt.show(fig1)
        
    # plot generated images
    generated_images = generate()
    generated_images = generated_images.detach().numpy()
    
    fig, ax = plt.subplots(8,8)
    fig.set_size_inches(12,12)
    fig.suptitle('Generated samples', fontsize=20)

    for i in range(8):
        for j in range(8):
            ax[i,j].imshow(generated_images[8*i+j].reshape(28,28), cmap='gray_r', vmin=0, vmax=1, interpolation='catrom')
            ax[i,j].axis('off')
    plt.show(block=True)
    
