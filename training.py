from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from data_generator import *
from plots import *
from networks import *


my_device = 'mps'
torch.set_default_device(my_device)

def train_network(config_name, process ,S_0, SDE_params, dts, T, use_Z, number_data_points, epochs, batch_size, advancing_C, log_freq):
    """
    -> config_name          -> string (the name used which the models would be saved) / check parameters.yaml
    -> process              -> string (wither GBM or CIR)
    -> S_0                  -> int initial value for the training data generation
    -> SDE_params           -> mu, sigma / kappa, s_bar, gamma for the training data generation
    -> dts                  -> list of integers, e.g. [0.05, 0.1, 0.2, 0.4, 0.5, 0.67, 1, 2]
    -> T                    -> int time to maturity needed for the training data generation
    -> use_Z                -> boolean -> false and you get vanilla conditional GAN, true and you get the supervised version
    -> number_data_points   -> int length of training data
    -> epochs               -> int 
    -> batch_size           -> int
    -> advancing_C          -> int
    -> log_freq             -> int
    """

    print('\033[35m', "-> Starting with the data generation \n")
    ### Training data ###
    list_training_data_tuples = generate_training_data(process, S_0, SDE_params, dts, number_data_points, T)
    # the list above has tuples (S_k, S_k+1, Z_dt)
    print('\033[35m', "-> Finished with data generation \n", '\033[30m')

    training_data_tensor = torch.tensor(list_training_data_tuples, dtype=torch.float32).to(device=my_device)
    dataset = TensorDataset(training_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    ### NN initialization ###
    generator = Generator(c = 2).to(device=torch.device(my_device))
    if use_Z :
        discriminator = Discriminator(c = 3).to(device=torch.device(my_device))
    else :
        discriminator = Discriminator(c = 2).to(device=torch.device(my_device))

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 5 * 1e-4, betas=(0.5, 0.999)) 
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 1e-4, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    ### Variables to keep track of training ###
    start_training_time = time.time()
    D_losses = []
    G_losses = []
    iteration = 0

    print('\033[36m', "-> Starting with model training \n")
    for epoch in range(epochs):
        for batch in dataloader:
            real_data = batch[0].to(my_device)

            first_steps = real_data[:, 0].view(1, -1)
            second_steps = real_data[:, 1].view(1, -1)
            Z = real_data[:, 2].view(1, -1)
            dts = real_data[:, 3].view(1, -1)

            real_labels = torch.ones(batch_size, 1).to(device=my_device)
            fake_labels = torch.zeros(batch_size, 1).to(device=my_device)

            ### training the discriminator ###
            discriminator.train()
            generator.eval()

            for _ in range(advancing_C):
                ### real ###
                conditions = (first_steps, dts, Z) if use_Z else (first_steps, dts)
                D_pred_real = discriminator.forward(second_steps, conditions)
                D_loss_real = loss_function(D_pred_real, real_labels)

                ### fake ###
                noise = Z if use_Z else torch.randn(batch_size, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
                fake_data = generator.forward(noise, (first_steps, dts)).view(1, -1)
                D_pred_fake = discriminator.forward(fake_data, conditions)
                D_loss_fake = loss_function(D_pred_fake, fake_labels)

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

            noise = Z if use_Z else torch.randn(batch_size, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
            
            fake_data = generator.forward(noise, (first_steps, dts)).view(1, -1)
            D_pred_fake = discriminator.forward(fake_data, conditions) 
            G_loss = loss_function(D_pred_fake, real_labels)
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

    losses(D_losses, G_losses, config_name)

    return D_losses, G_losses
