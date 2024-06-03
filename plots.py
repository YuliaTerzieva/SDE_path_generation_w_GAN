import matplotlib.pyplot as plt
import numpy as np
import torch
from data_generator import *

def losses(D_losses, G_losses, config_name):
    plt.figure()
    plt.plot(D_losses, label = "Discriminator loss")
    plt.plot(G_losses, label = "Generator loss")
    plt.legend()
    plt.savefig(f"Plots/Loss {config_name}.png")
    plt.show()

def log_returns(actual_log_returns, model, n_steps, n_paths, batch_size, c_dt, config_name, my_device = 'mps'):
    """
    actual_log_returns -> tensor 
    model -> torch model
    n_steps -> int
    n_paths -> int
    """
    pred_Log_Return = np.zeros((n_steps, n_paths))
    for i in range(n_steps):
        c_previous = actual_log_returns[i].view(1, -1)
        x_random = torch.randn(batch_size, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
        x_fake = model.forward(x_random, (c_previous, c_dt)).view(1, -1)
        pred_Log_Return[i] = x_fake.cpu().detach().numpy()

    plt.figure()
    plt.hist(actual_log_returns.cpu().detach().numpy().flatten(), bins = 50, alpha = 0.8, density=True, color = 'lightblue', label="Real")
    plt.hist(pred_Log_Return.flatten(), bins = 50, alpha = 0.5, density = True, color = 'palevioletred', label="Generated")
    plt.legend()
    plt.title(f"Log Returns {config_name}")
    plt.savefig(f"Plots/Log Returns {config_name}")
    plt.show()

def weak_stong_error_gen_paths(config_name, S_0 = 1, mu = 0.05, sigma = 0.2, n_steps_array = [50, 100, 200, 400], n_paths = 500, dt = 0.05, use_Z = False):
    np.random.seed(42)

    error_Weak_one_step = np.zeros((len(n_steps_array)))
    error_Strong_one_step = np.zeros((len(n_steps_array)))
    error_Weak_self_gen = np.zeros((len(n_steps_array)))
    error_Strong_self_gen = np.zeros((len(n_steps_array)))

    for i, n_steps in enumerate(n_steps_array):
        S_E, S_M, Exact_solution, Z, Log_Return = gen_paths_GBM(S_0, mu, sigma, dt, n_steps, n_paths)

        gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
        gen_model.eval()

        if use_Z :
            model_paths_one_step = gen_paths_from_GAN(gen_model, S_0, dt, n_steps, n_paths, actual_log_returns=Log_Return, Z_BM=Z)
            model_paths_self_gen =  gen_paths_from_GAN(gen_model, S_0, dt, n_steps, n_paths, Z_BM=Z)
        else :
            model_paths_one_step = gen_paths_from_GAN(gen_model, S_0, dt, n_steps, n_paths, actual_log_returns=Log_Return)
            model_paths_self_gen =  gen_paths_from_GAN(gen_model, S_0, dt, n_steps, n_paths)
        
        error_Weak_one_step[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(model_paths_one_step[-1]))
        error_Strong_one_step[i] = np.mean(np.abs(Exact_solution[-1]-model_paths_one_step[-1]))
        error_Weak_self_gen[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(model_paths_self_gen[-1]))
        error_Strong_self_gen[i] = np.mean(np.abs(Exact_solution[-1]-model_paths_self_gen[-1]))

    plt.plot(n_steps_array, error_Weak_one_step, linestyle='dashed', label = 'Weak error one step')
    plt.plot(n_steps_array, error_Strong_one_step, linestyle='dashed', label = 'Strong error one step')
    plt.plot(n_steps_array, error_Weak_self_gen, label = 'Weak error self gen')
    plt.plot(n_steps_array, error_Strong_self_gen, label = 'Strong error self gen')
    plt.xlabel("number of steps")
    plt.title(f"Plots/Weak_Stong {config_name}")
    plt.legend()
    plt.savefig(f"Plots/Weak_Stong {config_name}")
    plt.show()