import numpy as np
import matplotlib.pyplot as plt
import torch
import pdb

############## GBM process ##############

def gen_paths_GBM(S_0, mu, sigma, dt, n_steps, n_paths): 
    # note those equations are from the book Mathematical Modeling and Computation in Finance
    
    Z = np.random.normal(0, 1, [n_steps,n_paths])
    S_E = np.zeros((n_steps, n_paths)) # Euler
    S_M = np.zeros((n_steps, n_paths)) # Mistein 
    Log_Return = np.zeros((n_steps, n_paths))
    Exact_solution = np.zeros((n_steps, n_paths)) # Exact
    S_E[0, :] = S_M[0, :] = Exact_solution[0, :] = S_0

    for step_inx in range(n_steps-1):
        if n_paths > 1 :
            Z[step_inx, :] = (Z[step_inx, :] - np.mean(Z[step_inx,:])) / np.std(Z[step_inx,:])

        dW = (dt**(1/2))*Z[step_inx, :]
        S_E[step_inx+1, :] = S_E[step_inx, :] + S_E[step_inx, :] * mu * dt + S_E[step_inx, :] * sigma * dW # Equation 9.16

        S_M[step_inx+1, :] = S_M[step_inx, :] + S_M[step_inx, :] * mu * dt + S_M[step_inx, :] * sigma * dW\
                             + 0.5 * sigma**2 * S_M[step_inx, :] * (np.power(dW,2) - dt) # Equation between 9.17 and 9.18 :D
        
        Exact_solution[step_inx+1, :] = Exact_solution[step_inx, :] * np.exp((mu - 0.5 * sigma**2) *dt + sigma * dW) # Equation 9.17

        Log_Return[step_inx+1, :] = np.log(np.divide(Exact_solution[step_inx+1, :], Exact_solution[step_inx, :]))

    return S_E, S_M, Exact_solution, Z, Log_Return

############## CIR process ##############

def CIR_sample(kappa,gamma,vbar,s,t,v_s, n_paths): # from book equation 9.30
    delta = ( 4.0 * kappa * vbar ) / (gamma ** 2)
    c = (gamma**2 * (1-np.exp(-kappa*(t-s)))) / (4 * kappa)
    kappaBar = 4 * kappa * np.exp(-kappa*(t-s)) * v_s / (gamma**2 * (1 - np.exp(-kappa*(t-s))))
    sample = c * np.random.noncentral_chisquare(delta,kappaBar,n_paths)
    return  sample

def gen_paths_CIR(S_0, kappa, S_bar, gamma, dt, n_steps, n_paths):  
    
    Z = np.random.normal(0, 1, [n_steps,n_paths])
    S_E = np.zeros((n_steps, n_paths)) # Euler
    S_M = np.zeros((n_steps, n_paths)) # Mistein 
    Return = np.zeros((n_steps, n_paths))
    Exact_solution = np.zeros((n_steps, n_paths)) # Exact
    S_E[0, :] = S_M[0, :] = Exact_solution[0, :] = S_0
    Return[0] = (Exact_solution[0, :] - S_bar) / S_bar

    for step_inx in range(n_steps-1):
        if n_paths > 1 :
            Z[step_inx, :] = (Z[step_inx, :] - np.mean(Z[step_inx,:])) / np.std(Z[step_inx,:])

        dW = (dt**(1/2))*Z[step_inx, :]
        S_E[step_inx+1, :] =  S_E[step_inx, :] + kappa * (S_bar - S_E[step_inx, :]) * dt + gamma * np.sqrt(S_E[step_inx+1, :]) * dW 
        S_E[step_inx+1, :] = np.maximum(S_E[step_inx+1, :], 0)

        S_M[step_inx+1, :] = None
        
        Exact_solution[step_inx+1, :] = CIR_sample(kappa,gamma,S_bar, 0 , dt , Exact_solution[step_inx, :], n_paths)

        Return[step_inx+1, :] = (Exact_solution[step_inx+1, :] - S_bar ) / S_bar

    return S_E, S_M, Exact_solution, Z, Return

def gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt, n_steps, n_paths, actual_returns = None, Z_BM = None, my_device = 'mps'): 
    """
        gen_model -> torch model
        process -> sting 'GBM' or 'CIR'
        S_0, S_bar, dt, n_steps, n_paths -> ints
        actual_returns -> numpy array
        Z_BM -> numpy array
        my_device -> string
    """

    if Z_BM is not None :
        Z = torch.from_numpy(Z_BM).type(torch.FloatTensor).to(device=torch.device(my_device))
    else:
        Z = torch.randn((n_steps, n_paths)).type(torch.FloatTensor).to(device=torch.device(my_device))
        
    c_dt = torch.full((1, n_paths), dt).type(torch.FloatTensor).to(device=torch.device(my_device))
    G_paths = torch.zeros((n_steps, n_paths)).type(torch.FloatTensor).to(device=torch.device(my_device))
    G_paths[0, :] = S_0

    if actual_returns is not None:
        actual_returns = torch.from_numpy(actual_returns).type(torch.FloatTensor).to(device=torch.device(my_device))

    for step_inx in range(n_steps-1):
        if actual_returns is not None :
            input = actual_returns[step_inx].view(1, -1)
        else :
            input = G_paths[step_inx].view(1, -1)

        if process == 'GBM':
            gen_pred = gen_model(Z[step_inx].view(1, -1), (input, c_dt)).T
            gen_pred = torch.exp(gen_pred)
            gen_pred = gen_pred * G_paths[step_inx]

        elif process == 'CIR' : 
            gen_pred = gen_model(Z[step_inx].view(1, -1), (input, c_dt)).T
            gen_pred = gen_pred * S_bar
            gen_pred = torch.abs(gen_pred)

        G_paths[step_inx+1] = torch.squeeze(gen_pred)

    return G_paths.cpu().detach().numpy()

if __name__ == '__main__':

    config_name = 'supervised_GAN_CIR_config5'
    process = 'CIR'

    paths = 1

    S_E, S_M, Exact_solution, Z, Returns = gen_paths_CIR(0.1, 0.1, 0.1, 0.3, 0.05, 100, paths)
    gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
    gen_model.eval()        
    model_paths_one_step = gen_paths_from_GAN(gen_model, process, 0.1, 0.1, 0.05, 100, paths, actual_returns=Returns, Z_BM=Z)
    
    plt.plot(Exact_solution, linestyle='dashed', color = 'lightblue')
    plt.plot([], [], color = 'lightblue', label = "Exact")
    plt.plot(model_paths_one_step, color = 'palevioletred')
    plt.plot([], [], color = 'palevioletred', label = "Generated one-step")
    plt.title("Generated vs actual paths")
    plt.legend()
    plt.show()

    


