import matplotlib.pyplot as plt
import numpy as np
import torch

def losses(D_losses, G_losses, config_name):
    plt.figure()
    plt.plot(D_losses, label = "Discriminator loss")
    plt.plot(G_losses, label = "Generator loss")
    plt.legend()
    plt.savefig(f"Loss {config_name}.png")

def log_returns(actual_log_returns, model, n_steps, n_paths, batch_size, c_dt, config_name, my_device = 'mps'):
    """
    actual_log_returns -> tensor 
    model -> torch model
    n_steps -> int
    n_paths ->int
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
    plt.savefig(f"Log Returns {config_name}")