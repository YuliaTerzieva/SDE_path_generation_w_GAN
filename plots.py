import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import torch
from data_generator import *

def losses(D_losses, G_losses, config_name):
    plt.figure()
    plt.plot(D_losses, label = "Discriminator loss")
    plt.plot(G_losses, label = "Generator loss")
    plt.legend()
    plt.savefig(f"Plots/Loss/Loss {config_name}.png")
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
    plt.savefig(f"Plots/Log Returns/Log Returns {config_name}")
    plt.show()

def ECDF_multiple_dts(process, SDE_params, use_Z):
    S_t = 0.1
    dts = [0.05, 0.1, 0.5, 1, 2]
    paths = 1000

    gen_model = torch.load('Trained_Models/generator_scGAN_CIR_multiple_dts.pth') if use_Z else torch.load('Trained_Models/generator_cGAN_CIR_multiple_dts.pth')
    gen_model.eval()

    for dt in dts:
        steps = 100#int(2/dt) +1 
        c_dt = torch.full((1, paths), dt).type(torch.FloatTensor).to(device='mps')
        if process == 'GBM':
            _, _, Exact_solution, Z, Returns = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, paths, steps)
        else :
            _, _, Exact_solution, Z, Returns = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, paths, steps)

        Returns_torch = torch.from_numpy(Returns).type(torch.FloatTensor).to(device=torch.device('mps')).T
        Z_torch =  torch.from_numpy(Z).type(torch.FloatTensor).to(device=torch.device('mps')).T
        model_path = torch.full((steps, paths), S_t)
        for i in range(1,steps):
            c_previous = Returns_torch[i-1].view(1, -1)
            x_random = Z_torch[i-1].view(1, -1) if use_Z else torch.randn(paths, 1).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
            x_fake = gen_model.forward(x_random, (c_previous, c_dt)).view(1, -1)
            if process == 'GBM':
                model_path[i] = GBM_St_from_Rt(x_fake, model_path[i-1]).squeeze()
            else :
                model_path[i] = CIR_St_from_Rt(x_fake, SDE_params['S_bar']).squeeze()

        # ecdf = ECDF(Exact_solution.flatten())
        # plt.plot(ecdf.x, ecdf.y, color = "black", linestyle='dashed')
        ecdf = ECDF(model_path.cpu().detach().numpy().flatten())
        plt.plot(ecdf.x, ecdf.y, label = f'$\Delta t = {dt}$')
    
    plt.legend()
    plt.xlabel("$S_{t+\Delta t} | S_t$")
    plt.title("Supervised GAN") if use_Z else plt.title("Vanilla GAN")
    # plt.savefig()
    plt.xlim([0, 0.3])
    plt.show()

def weak_stong_error_gen_paths(config_name, process, S_0, SDE_params, n_steps_array, n_paths, dt, use_Z = False):
    np.random.seed(42)

    error_Weak_one_step = np.zeros((len(n_steps_array)))
    error_Strong_one_step = np.zeros((len(n_steps_array)))
    error_Weak_self_gen = np.zeros((len(n_steps_array)))
    error_Strong_self_gen = np.zeros((len(n_steps_array)))

    for i, n_steps in enumerate(n_steps_array):
        if process == 'GBM':
            _, _, Exact_solution, Z, Returns = gen_paths_GBM(S_0, SDE_params['mu'], SDE_params['sigma'], dt, n_steps, n_paths)
            S_bar = None
        elif process == 'CIR' : 
            _, _, Exact_solution, Z, Returns = gen_paths_CIR(S_0, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, n_steps, n_paths)
            S_bar = SDE_params['S_bar']

        gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
        gen_model.eval()

        if use_Z :
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt, n_steps, n_paths, actual_returns=Returns, Z_BM=Z)
            model_paths_self_gen =  gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt, n_steps, n_paths, Z_BM=Z)
        else :
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt, n_steps, n_paths, actual_returns=Returns)
            model_paths_self_gen =  gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt, n_steps, n_paths)
        
        error_Weak_one_step[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(model_paths_one_step[-1]))
        error_Strong_one_step[i] = np.mean(np.abs(Exact_solution[-1]-model_paths_one_step[-1]))
        error_Weak_self_gen[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(model_paths_self_gen[-1]))
        error_Strong_self_gen[i] = np.mean(np.abs(Exact_solution[-1]-model_paths_self_gen[-1]))

    plt.plot(n_steps_array, error_Weak_one_step, linestyle='dashed', label = 'Weak error one step')
    plt.plot(n_steps_array, error_Strong_one_step, linestyle='dashed', label = 'Strong error one step')
    plt.plot(n_steps_array, error_Weak_self_gen, label = 'Weak error self gen')
    plt.plot(n_steps_array, error_Strong_self_gen, label = 'Strong error self gen')
    plt.xlabel("number of steps")
    plt.title(f"Weak_Stong {config_name}")
    plt.legend()
    plt.savefig(f"Plots/Weak_Stong {config_name}")
    plt.show()

def weak_stong_error_gen_paths_multiple_dt(config_name, process, S_0, SDE_params, n_paths, dt, use_Z = False):
    np.random.seed(42)

    possible_dt_steps = np.array([1, 2, 4, 8, 10, 20, 40])

    error_Weak_GAN = np.zeros((len(possible_dt_steps)))
    error_Strong_GAN = np.zeros((len(possible_dt_steps)))
    error_Weak_Euler = np.zeros((len(possible_dt_steps)))
    error_Strong_Euler = np.zeros((len(possible_dt_steps)))
    error_Weak_Milstain = np.zeros((len(possible_dt_steps)))
    error_Strong_Milstain = np.zeros((len(possible_dt_steps)))

    for i, n_dt_step in enumerate(possible_dt_steps):
        n_steps = int(2 / (dt*n_dt_step)) + 1 

        print("dt = ", dt*n_dt_step)
        print("n_steps", 2 / (dt*n_dt_step), n_steps)

        if process == 'GBM':
            Euler, Milstain, Exact_solution, Z, Returns = gen_paths_GBM(S_0, SDE_params['mu'], SDE_params['sigma'], dt*n_dt_step, n_steps, n_paths)
            S_bar = None
        elif process == 'CIR' : 
            Euler, Milstain, Exact_solution, Z, Returns = gen_paths_CIR(S_0, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt*n_dt_step, n_steps, n_paths)
            S_bar = SDE_params['S_bar']

        gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
        gen_model.eval()

        if use_Z :
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt*n_dt_step, n_steps, n_paths, actual_returns=Returns, Z_BM=Z)
        else :
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt*n_dt_step, n_steps, n_paths, actual_returns=Returns)
        
        error_Weak_GAN[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(model_paths_one_step[-1]))
        error_Strong_GAN[i] = np.mean(np.abs(Exact_solution[-1]-model_paths_one_step[-1]))
        error_Weak_Euler[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(Euler[-1]))
        error_Strong_Euler[i] = np.mean(np.abs(Exact_solution[-1]-Euler[-1]))
        error_Weak_Milstain[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(Milstain[-1]))
        error_Strong_Milstain[i] = np.mean(np.abs(Exact_solution[-1]-Milstain[-1]))

    print(error_Weak_GAN)
    print(error_Strong_GAN)
    print(error_Weak_Euler)
    print(error_Strong_Euler)
    print(error_Weak_Milstain)
    print(error_Strong_Milstain)

    plt.plot(possible_dt_steps*dt, error_Weak_GAN, color = 'palevioletred', label = 'Weak error GAN')
    plt.plot(possible_dt_steps*dt, error_Strong_GAN, color = 'palevioletred', linestyle='dashed', label = 'Strong error GAN')
    plt.plot(possible_dt_steps*dt, error_Weak_Euler,  color = 'darkblue', label = 'Weak error Euler')
    plt.plot(possible_dt_steps*dt, error_Strong_Euler,  color = 'darkblue', linestyle='dashed', label = 'Strong error Euler')
    plt.plot(possible_dt_steps*dt, error_Weak_Milstain, color = 'lightblue', label = 'Weak error Milstain')
    plt.plot(possible_dt_steps*dt, error_Strong_Milstain, color = 'lightblue', linestyle='dashed', label = 'Strong error Milstain')
    plt.xlabel("$\delta t$")
    plt.title(f"Weak_Stong {config_name}")
    plt.legend()
    plt.savefig(f"Plots/Weak_Stong/Weak_Stong {config_name}")
    plt.show()

def ks_plot(config_name, process, SDE_params, use_Z):
    torch.random.manual_seed(42)
    np.random.seed(42)
    
    gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
    gen_model.eval()

    N_test = [100, 1_000, 10_000, 100_000]
    number_of_repetitions = 1

    KS = np.zeros((len(N_test), number_of_repetitions, 4))
    one_W = np.zeros((len(N_test), number_of_repetitions, 4))

    for count, N in enumerate(N_test):
        steps = 10
        paths = int(N/10)
        for rep in range(number_of_repetitions):
            if process == 'GBM':
                S_t = 0.1
                dt = 0.05
                Euler, Milstain, Exact_solution, Z, Returns = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, steps, paths)
                _, _, Exact_solution_2, _, _ = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, steps, paths)
            else :
                S_t = 0.4
                dt = 0.05
                Euler, Milstain, Exact_solution, Z, Returns = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, steps, paths)
                _, _, Exact_solution_2, _, _ = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, steps, paths)
            
            # print(Returns.shape)
            c_dt = torch.full((1, paths), dt).type(torch.FloatTensor).to(device='mps')
            Returns_torch = torch.from_numpy(Returns).type(torch.FloatTensor).to(device=torch.device('mps'))
            Z_torch =  torch.from_numpy(Z).type(torch.FloatTensor).to(device=torch.device('mps'))
            model_path = torch.full((steps, paths), S_t)
            for i in range(1,steps):
                c_previous = Returns_torch[i-1].view(1, -1)
                x_random = Z_torch[i-1].view(1, -1) if use_Z else torch.randn(paths, 1).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
                # print(c_previous.shape, x_random.shape, c_dt.shape) # should all be 1 x paths
                x_fake = gen_model.forward(x_random, (c_previous, c_dt)).view(1, -1)
                if process == 'GBM':
                    model_path[i] = GBM_St_from_Rt(x_fake, model_path[i-1]).squeeze()
                else :
                    model_path[i] = CIR_St_from_Rt(x_fake, SDE_params['S_bar']).squeeze()
            
            
            Exact_solution = Exact_solution[-1]
            Euler = Euler[-1]
            E_ks = stats.ks_2samp(Euler, Exact_solution)[0]
            E_one_W_distance = stats.wasserstein_distance(Euler, Exact_solution)

            Milstain = Milstain[-1]
            M_ks = stats.ks_2samp(Milstain, Exact_solution)[0]
            M_one_W_distance = stats.wasserstein_distance(Milstain, Exact_solution)

            model_path = model_path[-1].cpu().detach().numpy()
            GAN_ks = stats.ks_2samp(model_path, Exact_solution)[0]
            GAN_one_W_distance = stats.wasserstein_distance(model_path, Exact_solution)

            Exact_solution_2 = Exact_solution_2[-1]
            Exact_solution_ks = stats.ks_2samp(Exact_solution_2, Exact_solution)[0]
            Exact_solution_one_W_distance = stats.wasserstein_distance(Exact_solution_2, Exact_solution)

            KS[count, rep] = [E_ks, M_ks, GAN_ks, Exact_solution_ks]
            one_W[count, rep] = [E_one_W_distance, M_one_W_distance, GAN_one_W_distance, Exact_solution_one_W_distance]
        print(f"Done with N = {N}, repetition = {rep}")

    labels = ["Euler", 'Milstain', 'GAN', "Exact"]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    KS_means = KS.mean(axis=1)
    KS_std = KS.std(axis=1)
    for i in range(KS_means.shape[1]):
        ax1.fill_between(N_test, KS_means[:, i] - KS_std[:, i], KS_means[:, i] + KS_std[:, i], alpha=0.3)
        ax1.plot(N_test, KS_means[:, i], label = labels[i])
    
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("KS statistics")

    one_W_means = one_W.mean(axis=1)
    one_W_std = one_W.std(axis=1)
    for i in range(one_W_means.shape[1]):
        ax2.fill_between(N_test, one_W_means[:, i] - one_W_std[:, i], one_W_means[:, i] + one_W_std[:, i], alpha=0.3)
        ax2.plot(N_test, one_W_means[:, i], label = labels[i])

    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("1_Waserstain distance")

    fig.suptitle(f'Configuration {config_name}', fontsize=15)
    plt.savefig(f'Plots/KS_1W_{config_name}')
    plt.show()
