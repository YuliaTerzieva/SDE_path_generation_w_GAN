import numpy as np
import time
from data_generator import *
from plots import *
from networks import *

my_device = 'mps'
torch.set_default_device(my_device)

def train_network(config_name, process ,S_0, SDE_params, n_steps, n_paths, dt, number_data_points, epochs, batch_size, advancing_C, log_freq, use_Z, multiple_dt):

    if process == 'CIR' : 
    ### Generating the SDE paths ###
        _, _, _, Z, Returns = gen_paths_CIR(S_0, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, n_steps + 40, n_paths) # there is plus 40 in case we want the dt to be 2 :) 

    if process == 'GBM' : 
    ### Generating the SDE paths ###
        _, _, _, Z, Returns = gen_paths_GBM(S_0, SDE_params['mu'], SDE_params['sigma'], dt, n_steps + 40, n_paths) # there is plus 40 in case we want the dt to be 2 :) 

    ### NN initialization ###
    generator = Generator(c = 2).to(device=torch.device(my_device))
    discriminator = Discriminator(c = 2).to(device=torch.device(my_device))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 5 * 1e-4, betas=(0.5, 0.999)) 
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 1e-4, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    ### Making tensors ###
    if use_Z :
        Z_tensor = torch.from_numpy(Z).type(torch.FloatTensor).to(device=torch.device(my_device))
    Log_Return_tensor = torch.from_numpy(Returns).type(torch.FloatTensor).to(device=torch.device(my_device))

    
    possible_dt_steps = [1, 2, 4, 8, 10, 20, 40] if multiple_dt else [1]
    ### Variables to keep track of training ###
    start_training_time = time.time()
    D_losses = []
    G_losses = []
    iteration = 0

    for epoch in range(epochs):
        for i in range(0, number_data_points, batch_size):
            ### training the discriminator ###
            discriminator.train()
            generator.eval()

            ### Real data ###
            step_index = np.random.choice(n_steps, size=batch_size, replace=True)
            path_index = np.random.choice(n_paths, size=batch_size, replace=True)
            num_steps = np.random.choice(possible_dt_steps, size=batch_size, replace=True) # this corresponds to dt [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2]
            input_real = Log_Return_tensor[[step_index+num_steps], [path_index]]
            c_previous = Log_Return_tensor[[step_index], [path_index]]
            c_dt = torch.full((1, batch_size), dt).type(torch.FloatTensor).to(device=my_device) * torch.tensor(num_steps, dtype=torch.float32)
            
            for _ in range(advancing_C):
                D_pred_real = discriminator.forward(input_real, (c_previous, c_dt))
                D_loss_real = loss_function(D_pred_real, torch.ones(D_pred_real.size(0), 1))
        
                ### Generator noise ###
                if use_Z : 
                    if multiple_dt:
                        # Initialize gen_input_noise
                        gen_input_noise = torch.zeros(batch_size, 1)
                        for i in range(batch_size):
                            start_index = step_index[i]
                            end_index = step_index[i] + num_steps[i]
                            gen_input_noise[i] = torch.sum(Z_tensor[start_index:end_index, path_index[i]])
                        gen_input_noise = gen_input_noise.view(1, -1)
                    else:  
                        gen_input_noise = Z_tensor[[step_index], [path_index]]
                else:
                    gen_input_noise = torch.randn(batch_size, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)

                x_fake = generator.forward(gen_input_noise, (c_previous, c_dt)).T
                D_pred_fake = discriminator.forward(x_fake, (c_previous, c_dt))
                D_loss_fake = loss_function(D_pred_fake, torch.zeros(D_pred_fake.size(0), 1))
        
                ### Combined losses ###
                D_loss = D_loss_real + D_loss_fake
                
                ### Learning ###
                discriminator_optimizer.zero_grad()
                D_loss.backward()
                discriminator_optimizer.step()

            D_losses.append(D_loss.item())

            ### training the generator ###
            discriminator.eval()
            generator.train()

            x_fake = generator.forward(gen_input_noise, (c_previous, c_dt)).T
            D_pred_fake = discriminator.forward(x_fake, (c_previous, c_dt))
            G_loss = loss_function(D_pred_fake, torch.ones(D_pred_fake.size(0), 1))
            G_losses.append(G_loss.item())

            ### Learning ###
            generator_optimizer.zero_grad()
            G_loss.backward()
            generator_optimizer.step()
        
            ### update learning rate ###
            iteration +=1
            if iteration % 500 == 0 :
                generator_optimizer.param_groups[0]['lr'] /= 1.05

        ###  Log  ###
        if epoch % log_freq == 0:
            print(f"Epoch {epoch} -> D loss {D_loss.item()} -> G loss {G_loss.item()}")

    stop_training_time = time.time()
    print(f"Duration of training : {(stop_training_time - start_training_time)/60} in minutes")

    torch.save(generator, f"Trained_Models/generator_{config_name}.pth")
    torch.save(discriminator, f"Trained_Models/discriminator_{config_name}.pth")

    log_returns(Log_Return_tensor, generator, n_steps, n_paths, batch_size, c_dt, config_name, my_device)
    losses(D_losses, G_losses, config_name)

    return D_losses, G_losses
