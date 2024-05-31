import numpy as np
from data_generator import gen_paths_GBM
from plots import *
from vanilla_networks import *
import matplotlib.pyplot as plt
import pdb
import time


my_device = 'mps'
torch.set_default_device(my_device)

### Parameters ###
config_name = "cGAN_one_dt_mu_0_05_sigma_0_2"
S_0 = 1
mu = 0.05
sigma = 0.2
n_steps = 100
n_paths = 1000
dt = 0.05
number_data_points = 100_000
epochs = 200
batch_size = 1000
advancing_C = 2
log_freq = 10

### Generating the SDE paths ###
GBM_Euler, GBM_Milstein, GBM_Exact, Z, Log_Return = gen_paths_GBM(S_0, mu, sigma, dt, n_steps, n_paths)

### NN initialization ###
generator = Generator(c = 2).to(device=torch.device(my_device))
discriminator = Discriminator(c = 2).to(device=torch.device(my_device))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 5 * 1e-4, betas=(0.5, 0.999)) 
generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 1e-4, betas=(0.5, 0.999))
loss_function = nn.BCELoss()

# Turning then into tensors
Log_Return_tensor = torch.from_numpy(Log_Return).type(torch.FloatTensor).to(device=torch.device(my_device))
c_dt = torch.full((1, batch_size), dt).type(torch.FloatTensor).to(device=torch.device(my_device))

# Variables to keep track of training
start_training_time = time.time()
D_losses = []
G_losses = []
iteration = 0

for epoch in range(epochs):
    for i in range(0, number_data_points, batch_size):
        ### training the discriminator ###
        discriminator.train()
        generator.eval()

        # Real data #
        step_index = np.random.choice(n_steps, size=batch_size, replace=True)
        path_index = np.random.choice(n_paths, size=batch_size, replace=True)
        input_real = Log_Return_tensor[[step_index+1], [path_index]]
        c_previous = Log_Return_tensor[[step_index], [path_index]]
        
        for _ in range(advancing_C):
            D_pred_real = discriminator.forward(input_real, (c_previous, c_dt))
            D_loss_real = loss_function(D_pred_real, torch.ones(D_pred_real.size(0), 1))
    
            # Fake data #
            x_random = torch.randn(batch_size, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)

            x_fake = generator.forward(x_random, (c_previous, c_dt)).T
            D_pred_fake = discriminator.forward(x_fake, (c_previous, c_dt))
            D_loss_fake = loss_function(D_pred_fake, torch.zeros(D_pred_fake.size(0), 1))
    
            # Combined #
            D_loss = D_loss_real + D_loss_fake
            
            # Learning #
            discriminator_optimizer.zero_grad()
            D_loss.backward()
            discriminator_optimizer.step()

        D_losses.append(D_loss.item())

        ### training the generator ###
        discriminator.eval()
        generator.train()

        x_random = torch.randn(batch_size, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)

        x_fake = generator.forward(x_random, (c_previous, c_dt)).T
        D_pred_fake = discriminator.forward(x_fake, (c_previous, c_dt))
        G_loss = loss_function(D_pred_fake, torch.ones(D_pred_fake.size(0), 1))
        G_losses.append(G_loss.item())

        # Learning #
        generator_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()
    
        ## update learning rate ###
        iteration +=1
        if iteration % 500 == 0 :
            generator_optimizer.param_groups[0]['lr'] /= 1.05

    #  Log  #
    if epoch % log_freq == 0:
        print(f"Epoch {epoch} -> D loss {D_loss.item()} -> G loss {G_loss.item()}")

stop_training_time = time.time()

torch.save(generator, f"generator_{config_name}.pth")
torch.save(discriminator, f"discriminator_{config_name}.pth")

print(f"Duration of training : {stop_training_time - start_training_time} in seconds")

losses(D_losses, G_losses, config_name)

log_returns(Log_Return_tensor, generator, n_steps, n_paths, batch_size, c_dt, config_name, my_device)