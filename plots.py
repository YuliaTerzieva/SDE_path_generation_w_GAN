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

def ECDF_plot(gen_model, config_name, process, S_t, SDE_params, use_Z,  my_device = 'mps'):
    dts = [0.1, 0.5, 1, 2]
    points = 10_000
    first_step = torch.full((points, ), S_t).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)

    gen_model.eval()

    for dt in dts:
        if process == 'GBM':
            _, _, _, Z, Returns = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, 2, points)
        else :
            _, _, _, Z, Returns = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, 2, points)

        ecdf = ECDF(Returns[-1])
        plt.plot(ecdf.x, ecdf.y, color = "black", linestyle='dashed')
        
        Z = torch.tensor(Z[-1], dtype=torch.float32).to(device=torch.device(my_device)).view(1, -1)
        noise = Z if use_Z else torch.randn(points, 1).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
        
        dt_torch = torch.full((points, ), dt).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
        if process == 'GBM':
            model_dist = gen_model(noise, [dt_torch])
        elif process == 'CIR':
            model_dist = gen_model(noise, [first_step, dt_torch])
        ecdf = ECDF(model_dist.cpu().detach().numpy().flatten())
        plt.plot(ecdf.x, ecdf.y, label = f'$\Delta t = {dt}$')
    
    plt.legend()
    plt.xlabel("$S_{t+\Delta t} | S_t$")
    plt.title("Supervised GAN") if use_Z else plt.title("Vanilla GAN")
    plt.savefig(f"Plots/ECDF/ECDF_{config_name}")
    plt.show()

def ks_plot(gen_model, config_name, process, S_t, SDE_params, use_Z, my_device = 'mps'):
    torch.random.manual_seed(42)
    np.random.seed(42)
    
    N_test = [100, 300, 600, 1_000, 3_000, 6_000, 10_000, 30_000, 60_000, 100_000]
    number_of_repetitions = 10

    KS = np.zeros((len(N_test), number_of_repetitions, 4))
    one_W = np.zeros((len(N_test), number_of_repetitions, 4))

    dt = 0.4
    
    for count, N in enumerate(N_test):
        for rep in range(number_of_repetitions):
            Euler, Milstain, Exact_solution, Exact_solution_2, model_path, _ = dist_stock_step(gen_model, process, S_t, SDE_params, dt, 2, N, use_Z)
            
            Exact_solution = Exact_solution[-1]
            Euler = Euler[-1]
            E_ks = stats.ks_2samp(Euler, Exact_solution)[0]
            E_one_W_distance = stats.wasserstein_distance(Euler, Exact_solution)

            Milstain = Milstain[-1]
            M_ks = stats.ks_2samp(Milstain, Exact_solution)[0]
            M_one_W_distance = stats.wasserstein_distance(Milstain, Exact_solution)

            GAN_ks = stats.ks_2samp(model_path, Exact_solution)[0]
            GAN_one_W_distance = stats.wasserstein_distance(model_path, Exact_solution)

            Exact_solution_2 = Exact_solution_2[-1]
            Exact_solution_ks = stats.ks_2samp(Exact_solution_2, Exact_solution)[0]
            Exact_solution_one_W_distance = stats.wasserstein_distance(Exact_solution_2, Exact_solution)

            KS[count, rep] = [E_ks, M_ks, GAN_ks, Exact_solution_ks]
            one_W[count, rep] = [E_one_W_distance, M_one_W_distance, GAN_one_W_distance, Exact_solution_one_W_distance]

    labels = ["Euler", 'Milstain', 'GAN', "Exact"]


    KS_means = KS.mean(axis=1)
    KS_std = KS.std(axis=1)
    for i in range(KS_means.shape[1]):
        plt.fill_between(N_test, KS_means[:, i] - KS_std[:, i], KS_means[:, i] + KS_std[:, i], alpha=0.3)
        plt.plot(N_test, KS_means[:, i], label = labels[i])
    
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title("KS statistics")
    plt.savefig(f'Plots/KS_1W/KS_{config_name}')
    plt.show()
    

    one_W_means = one_W.mean(axis=1)
    one_W_std = one_W.std(axis=1)
    for i in range(one_W_means.shape[1]):
        plt.fill_between(N_test, one_W_means[:, i] - one_W_std[:, i], one_W_means[:, i] + one_W_std[:, i], alpha=0.3)
        plt.plot(N_test, one_W_means[:, i], label = labels[i])

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title("1_Waserstain distance")
    plt.savefig(f'Plots/KS_1W/1W_{config_name}')
    plt.show()

def weak_stong_error(gen_model, config_name, process, S_t, SDE_params, dts, T, n_paths, use_Z):
    np.random.seed(42)

    error_Weak_GAN = np.zeros((len(dts)))
    error_Strong_GAN = np.zeros((len(dts)))
    error_Weak_Euler = np.zeros((len(dts)))
    error_Strong_Euler = np.zeros((len(dts)))
    error_Weak_Milstain = np.zeros((len(dts)))
    error_Strong_Milstain = np.zeros((len(dts)))

    for i, dt in enumerate(dts):
        n_steps = int(T/dt)+1
        if process == 'GBM':
            Euler, Milstain, Exact_solution, Z, Returns = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, n_steps, n_paths)
            S_bar = None
        elif process == 'CIR' : 
            Euler, Milstain, Exact_solution, Z, Returns = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, n_steps, n_paths)
            S_bar = SDE_params['S_bar']

        gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
        gen_model.eval()

        if use_Z :
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_t, S_bar, dt, n_steps, n_paths, actual_returns=Returns, Z_BM=Z)
        else :
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_t, S_bar, dt, n_steps, n_paths, actual_returns=Returns)
        
        error_Weak_GAN[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(model_paths_one_step[-1]))
        error_Strong_GAN[i] = np.mean(np.abs(Exact_solution[-1]-model_paths_one_step[-1]))
        error_Weak_Euler[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(Euler[-1]))
        error_Strong_Euler[i] = np.mean(np.abs(Exact_solution[-1]-Euler[-1]))
        error_Weak_Milstain[i] = np.abs(np.mean(Exact_solution[-1])-np.mean(Milstain[-1]))
        error_Strong_Milstain[i] = np.mean(np.abs(Exact_solution[-1]-Milstain[-1]))

    plt.plot(dts, error_Weak_GAN, color = 'palevioletred', label = 'Weak error GAN')
    plt.plot(dts, error_Strong_GAN, color = 'palevioletred', linestyle='dashed', label = 'Strong error GAN')
    plt.plot(dts, error_Weak_Euler,  color = 'darkblue', label = 'Weak error Euler')
    plt.plot(dts, error_Strong_Euler,  color = 'darkblue', linestyle='dashed', label = 'Strong error Euler')
    plt.plot(dts, error_Weak_Milstain, color = 'lightblue', label = 'Weak error Milstain')
    plt.plot(dts, error_Strong_Milstain, color = 'lightblue', linestyle='dashed', label = 'Strong error Milstain')
    plt.xlabel("$\Delta t$")
    plt.title(f"Weak_Stong {config_name}")
    plt.legend()
    plt.savefig(f"Plots/Weak_Strong/Weak_Strong {config_name}")
    plt.show()

def supervised_vs_not_generator_map():

    dt = 1 # he uses 1
    S_0 = 0.1
    _, _, _, Z, Returns = gen_paths_GBM(S_0, 0.05, 0.2, dt, 2, 100)

    plt.scatter(Z[-2], Returns[-1], color='black', alpha=0.5, s=20, label="Exact")

    Z_test = torch.randn(100, 1).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
    S_t = torch.full((100,), S_0).type(torch.FloatTensor).to(device='mps').view(1, -1)
    dt_tensor = torch.full((100,), dt).type(torch.FloatTensor).to(device='mps').view(1, -1)

    cGBM = torch.load(f'Trained_Models/generator_cGAN_GBM.pth') 
    cGBM.eval()
    scGBM = torch.load(f'Trained_Models/generator_scGAN_GBM.pth')
    scGBM.eval()

    plt.scatter(Z_test.cpu().detach().numpy(), cGBM.forward(Z_test, [dt_tensor]).cpu().detach().numpy(), color = 'lightblue', marker = "*", s=20, label = "conditional")
    plt.scatter(Z_test.cpu().detach().numpy(), scGBM.forward(Z_test, [dt_tensor]).cpu().detach().numpy(), color = 'orange', marker = "*", s=20, label = "supervised")
    plt.legend()
    plt.xlabel("Z")
    plt.ylabel("Generator output")
    plt.title("GBM")
    plt.show()

    _, _, _, Z, Returns = gen_paths_CIR(S_0, 0.1, 0.1, 0.1, dt, 2, 100)
    plt.scatter(Z[-2], Returns[-1], color = 'black', alpha=0.5, label = "Exact")
    cCIR = torch.load(f'Trained_Models/generator_cGAN_CIR_Feller.pth')
    cCIR.eval()
    scCIR = torch.load(f'Trained_Models/generator_scGAN_CIR_Feller.pth')
    scCIR.eval()

    plt.scatter(Z_test.cpu().detach().numpy(), cCIR.forward(Z_test, (S_t, dt_tensor)).cpu().detach().numpy(),  color = 'lightblue', marker = "*", s=20, label = "conditional")

    plt.scatter(Z_test.cpu().detach().numpy(), scCIR.forward(Z_test, (S_t, dt_tensor)).cpu().detach().numpy(), color = 'orange', marker = "*", s=20, label = "supervised")
    plt.legend()
    plt.xlabel("Z")
    plt.ylabel("Generator output")
    plt.title("CIR Feller satisfied")
    plt.show()

    _, _, _, Z, Returns = gen_paths_CIR(S_0, 0.1, 0.1, 0.3, dt, 2, 100)
    plt.scatter(Z[-2], Returns[-1], color = 'black', alpha=0.5, label = "Exact")
    cCIR = torch.load(f'Trained_Models/generator_cGAN_CIR_no_Feller.pth')
    cCIR.eval()
    scCIR = torch.load(f'Trained_Models/generator_scGAN_CIR_no_Feller.pth')
    scCIR.eval()

    plt.scatter(Z_test.cpu().detach().numpy(), cCIR.forward(Z_test, (S_t, dt_tensor)).cpu().detach().numpy(), color = 'lightblue', marker = "*", s=20, label = "conditional")

    plt.scatter(Z_test.cpu().detach().numpy(), scCIR.forward(Z_test, (S_t, dt_tensor)).cpu().detach().numpy(), color = 'orange', marker = "*", s=20, label = "supervised")
    plt.legend()
    plt.xlabel("Z")
    plt.ylabel("Generator output")
    plt.title("CIR Feller not satisfied")
    plt.show()

def discriminator_map(d_model, config_name, process, S_t, my_device = 'mps'):
    
    points = 1000

    dt = 1
    dt_torch = torch.full((points, ), dt).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
    S_t_torch = torch.full((points, ), S_t).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
     
    possible_Z =  torch.linspace(-3, 3, points).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
    possible_returns = np.linspace(-1.5, 3, points)
    map = torch.zeros(points, points).type(torch.FloatTensor).to(device=torch.device('mps'))
    for count, r in enumerate(possible_returns):
        r_torch = torch.full((points, ), r).type(torch.FloatTensor).to(device=torch.device(my_device)).view(1, -1)
        map[count] = d_model.forward(r_torch, (S_t_torch, dt_torch, possible_Z)).view(1, -1) if process == "CIR" else d_model.forward(r_torch, (dt_torch, possible_Z)).view(1, -1)


    plt.figure(figsize=(10, 6))
    plt.imshow(map.cpu().detach().numpy(), extent=(-3, 3, -1.5, 3), origin='lower', aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Model Output')
    plt.title(config_name)
    plt.xlabel("Z")
    plt.ylabel("Generator output")

    if process == 'GBM':
        _, _, _, Z, Returns = gen_paths_GBM(S_t, 0.05, 0.2, dt, 2, 100)
    
        plt.scatter(Z[-2], Returns[-1], color='black', alpha=0.5, s=20, label="Exact")

        Z_test = torch.randn(100, 1).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
        S_t = torch.full((100,), S_t).type(torch.FloatTensor).to(device='mps').view(1, -1)
        dt_tensor = torch.full((100,), dt).type(torch.FloatTensor).to(device='mps').view(1, -1)

        cGBM = torch.load(f'Trained_Models/generator_cGAN_GBM.pth') 
        cGBM.eval()
        scGBM = torch.load(f'Trained_Models/generator_scGAN_GBM.pth')
        scGBM.eval()

        plt.scatter(Z_test.cpu().detach().numpy(), cGBM.forward(Z_test, [dt_tensor]).cpu().detach().numpy(), color = 'lightblue', marker = "*", s=20, label = "conditional")
        plt.scatter(Z_test.cpu().detach().numpy(), scGBM.forward(Z_test, [dt_tensor]).cpu().detach().numpy(), color = 'orange', marker = "*", s=20, label = "supervised")
        plt.legend()
        plt.xlabel("Z")
        plt.ylabel("Generator output")
    
    else :
        _, _, _, Z, Returns = gen_paths_CIR(S_t, 0.1, 0.1, 0.3, dt, 2, 100) # Note if you do Feller condition change the gamma to 0.1
        plt.scatter(Z[-2], Returns[-1], color = 'black', alpha=0.5, s=20, label = "Exact")

        Z_test = torch.randn(100, 1).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
        S_t = torch.full((100,), S_t).type(torch.FloatTensor).to(device='mps').view(1, -1)
        dt_tensor = torch.full((100,), dt).type(torch.FloatTensor).to(device='mps').view(1, -1)

        cCIR = torch.load(f'Trained_Models/generator_cGAN_CIR_no_Feller.pth')# Note if you do Feller condition change the name of the file
        cCIR.eval()
        scCIR = torch.load(f'Trained_Models/generator_scGAN_CIR_no_Feller.pth')# Note if you do Feller condition change the name of the file
        scCIR.eval()

        plt.scatter(Z_test.cpu().detach().numpy(), cCIR.forward(Z_test, (S_t, dt_tensor)).cpu().detach().numpy(), color = 'lightblue', marker = "*", s=20, label = "conditional")

        plt.scatter(Z_test.cpu().detach().numpy(), scCIR.forward(Z_test, (S_t, dt_tensor)).cpu().detach().numpy(), color = 'orange', marker = "*", s=20, label = "supervised")
        plt.legend()
        plt.xlabel("Z")
        plt.ylabel("Generator output")
        
    plt.show()
