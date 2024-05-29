import numpy as np
from data_generator import gen_paths_GBM
from vanilla_networks import *
import matplotlib.pyplot as plt

my_device = 'mps'
torch.set_default_device(my_device)

### SDE parameters ###
S_0 = 1
mu = 0.05
sigma = 0.2
n_steps = 100
paths = 1000
T = 5
dt = 0.05

### Generating the SDE paths ###
GBM_Euler, GBM_Milstein, GBM_Exact, Z, Log_Return = gen_paths_GBM(S_0, mu, sigma, n_steps+1, paths, T)

### NN initialization ###
generator = Generator(c = 2).to(device=torch.device(my_device)) # conditional network with dt and S_k as conditions
discriminator = Discriminator(c = 2).to(device=torch.device(my_device))

### Training parameters ###
loss_function = nn.BCELoss()
number_data_points = 100_000
epochs = 200
batch_size = 1_000
advancing_C = 2
log_freq = 10

permutation_index_step = np.random.permutation(n_steps)

# ToDo : learning rate of the generator is decreased by a factor 1.05 every 500 iterations
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 1e-4, betas=(0.5, 0.999))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 1e-4, betas=(0.5, 0.999))


Log_Return_tensor = torch.from_numpy(Log_Return).type(torch.FloatTensor).to(device=torch.device(my_device))
c_dt = torch.full((1, paths), dt).type(torch.FloatTensor).to(device=torch.device(my_device))

D_losses = []
G_losses = []
for epoch in range(200):
    for i in range(0, number_data_points, batch_size):
        ### training the discriminator ###
        discriminator.train()
        generator.eval()

        # Real data #
        index = permutation_index_step[i%batch_size] # this has shape batch_size x 1
        input_real = Log_Return_tensor[index+1].view(1, -1) # this are all the paths are step index
        c_previous = Log_Return_tensor[index].view(1, -1) # passing S_k-1

        for _ in range(advancing_C):
            D_pred_real = discriminator.forward(input_real, (c_previous, c_dt))
            D_loss_real = loss_function(D_pred_real, torch.ones(D_pred_real.size(0), 1))
    
            # Fake data #
            x_random = torch.randn(paths, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)

            x_fake = generator.forward(x_random, (c_previous, c_dt)).view(1, -1)
            D_pred_fake = discriminator.forward(x_fake, (c_previous, c_dt))
            D_loss_fake = loss_function(D_pred_fake, torch.zeros(D_pred_fake.size(0), 1))
    
            # Combined #
            D_loss = D_loss_real + D_loss_fake
            D_losses.append(D_loss.item())
    
            # Learning #
            discriminator_optimizer.zero_grad()
            D_loss.backward()
            discriminator_optimizer.step()

        ### training the generator ###
        discriminator.eval()
        generator.train()

        x_random = torch.randn(paths, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
    
        x_fake = generator.forward(x_random, (c_previous, c_dt)).view(1, -1)
        D_pred_fake = discriminator.forward(x_fake, (c_previous, c_dt))
        G_loss = loss_function(D_pred_fake, torch.ones(D_pred_fake.size(0), 1))
        G_losses.append(G_loss.item())

        # Learning #
        generator_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()

    #  Log  #
    if epoch % log_freq == 0:
        print(f'Epoch: {epoch:2}  G loss: {G_loss.item():10.8f} D loss : {D_loss.item():10.8f}')

torch.save(generator, "generator_first.pth")
torch.save(discriminator, "discriminator_first.pth")

plt.plot(D_losses, label = "Discriminator loss")
plt.plot(G_losses, label = "Generator loss")
plt.legend()
plt.show()

