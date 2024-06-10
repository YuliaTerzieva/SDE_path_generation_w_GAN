import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import random

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

def CIR_sample(kappa, gamma, S_bar, dt, S_t, n_paths): # from book equation 9.30
    delta = ( 4.0 * kappa * S_bar ) / (gamma ** 2)
    c = (gamma**2 * (1-np.exp(-kappa*dt))) / (4 * kappa)
    kappaBar = (4 * kappa * np.exp(-kappa*dt) * S_t) / (gamma**2 * (1 - np.exp(-kappa*dt)))
    sample = c * np.random.noncentral_chisquare(delta,kappaBar,n_paths)
    return  sample

def gen_paths_CIR(S_0, kappa, S_bar, gamma, dt, n_steps, n_paths):  
    
    Z = np.random.normal(0, 1, [n_steps,n_paths])
    S_E = np.zeros((n_steps, n_paths)) # Euler
    S_M = np.zeros((n_steps, n_paths)) # Mistein 
    Return = np.zeros((n_steps, n_paths))
    Exact_solution = np.zeros((n_steps, n_paths)) # Exact
    S_E[0, :] = S_M[0, :] = Exact_solution[0, :] = S_0

    for step_inx in range(n_steps-1):
        if n_paths > 1 :
            Z[step_inx, :] = (Z[step_inx, :] - np.mean(Z[step_inx,:])) / np.std(Z[step_inx,:])

        dW = (dt**(1/2))*Z[step_inx, :]
        S_E[step_inx+1, :] =  S_E[step_inx, :] + kappa * (S_bar - S_E[step_inx, :]) * dt + gamma * np.sqrt(S_E[step_inx, :]) * dW 
        S_E[step_inx+1, :] = np.maximum(S_E[step_inx+1, :], 0)

        S_M[step_inx+1, :] = S_M[step_inx, :] + kappa * (S_bar - S_M[step_inx, :]) * dt + gamma * np.sqrt(S_M[step_inx, :]) * dW\
                             + 0.25 * gamma**2 * (np.power(dW,2) - dt)
        S_M[step_inx+1, :] = np.maximum(S_M[step_inx+1, :], 0)
                             
        Exact_solution[step_inx+1, :] = CIR_sample(kappa,gamma,S_bar, dt, Exact_solution[step_inx, :], n_paths)

        Return[step_inx+1, :] = (Exact_solution[step_inx+1, :] - S_bar ) / S_bar

    return S_E, S_M, Exact_solution, Z, Return

############## from R_t+1 | S_t to S_t+1 | S_t (returns to stock) ##############

def GBM_St_from_Rt(tensor, GBM_last_step):
    gen_pred = tensor
    gen_pred = torch.exp(gen_pred)
    gen_pred = torch.mul(gen_pred, GBM_last_step)
    return gen_pred

def CIR_St_from_Rt(tensor, S_bar):
    gen_pred = tensor
    gen_pred = gen_pred + 1
    gen_pred = gen_pred * S_bar
    gen_pred = torch.abs(gen_pred)
    return gen_pred

def dist_stock_step(gen_model, process, S_t, SDE_params, dt, steps, paths, use_Z):
    if process == 'GBM':
        Euler, Milstain, Exact_solution, Z, _ = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, steps, paths)
        _, _, Exact_solution_2, _, _ = gen_paths_GBM(S_t, SDE_params['mu'], SDE_params['sigma'], dt, steps, paths)
    else :
        Euler, Milstain, Exact_solution, Z, _ = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, steps, paths)
        _, _, Exact_solution_2, _, _ = gen_paths_CIR(S_t, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, steps, paths)
    
    c_dt = torch.full((paths, ), dt).type(torch.FloatTensor).to(device='mps').view(1, -1)
    first_step = torch.full((paths, ), S_t).type(torch.FloatTensor).to(device='mps').view(1, -1)
    Z_torch =  torch.from_numpy(Z[-1]).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)

    if process == 'GBM':
        x_random = Z_torch if use_Z else torch.randn(paths,).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
        x_fake = gen_model.forward(x_random, [c_dt])
        model_path = GBM_St_from_Rt(x_fake, S_t).cpu().detach().numpy().squeeze()
    elif process == 'CIR':
        x_random = Z_torch if use_Z else torch.randn(paths,).type(torch.FloatTensor).to(device=torch.device('mps')).view(1, -1)
        x_fake = gen_model.forward(x_random, [first_step, c_dt])
        model_path = CIR_St_from_Rt(x_fake, SDE_params['S_bar']).cpu().detach().numpy().squeeze()

    return Euler, Milstain, Exact_solution, Exact_solution_2, model_path, Z

############## gen training dataset ##############
def generate_training_data(process, S_0, SDE_params, dts, number_data_points, T):

    training_data = []
    for dt in dts:
        print('\033[35m', f"---> Starting generation of paths with dt of {dt}")
        n_steps = int(T/dt + 1)
        n_paths = 1_000

        if process == 'CIR' : 
            _, _, _, Z, Returns = gen_paths_CIR(S_0, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, n_steps + 1, n_paths) 

        if process == 'GBM' : 
            _, _, _, Z, Returns = gen_paths_GBM(S_0, SDE_params['mu'], SDE_params['sigma'], dt, n_steps + 1, n_paths)
        
        first_steps = Returns[:-1, :]  # All rows except the last one
        second_steps = Returns[1:, :]  # All rows except the first one
        corresponding_Z = Z[:-1, :]    # Corresponding Z values for the first steps

        corresponding_dt = np.full(first_steps.shape, dt)

        if process == 'GBM' :
            stacked_tuples = np.stack((second_steps, corresponding_Z, corresponding_dt), axis=-1)

        elif process == 'CIR':
            # Stack the arrays along a new axis to create tuples
            stacked_tuples = np.stack((first_steps, second_steps, corresponding_Z, corresponding_dt), axis=-1)

        # Flatten the stacked_tuples to a list of tuples
        tuple_list = [tuple(stacked_tuples[i, j]) for i in range(stacked_tuples.shape[0]) for j in range(stacked_tuples.shape[1])]

        # Sample 12,500 tuples with replacement
        sampled_tuples = random.choices(tuple_list, k=int(number_data_points / len(dts)))
        training_data.extend(sampled_tuples)
    
    random.shuffle(training_data)
    return training_data

############## Make stock ##############

def gen_paths_from_GAN(gen_model, process, S_0, S_bar, dt, n_steps, n_paths, actual_returns, Z_BM = None, my_device = 'mps'): 
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

    actual_returns = torch.from_numpy(actual_returns).type(torch.FloatTensor).to(device=torch.device(my_device))

    for step_inx in range(n_steps-1):
        if process == 'GBM':
            gen_pred = gen_model(Z[step_inx].view(1, -1), [c_dt]).T
            gen_pred = torch.exp(gen_pred)
            gen_pred = gen_pred * G_paths[step_inx]
            G_paths[step_inx+1] = torch.squeeze(gen_pred)

        elif process == 'CIR' : 
            input = actual_returns[step_inx].view(1, -1)
            gen_pred = CIR_St_from_Rt(gen_model(Z[step_inx].view(1, -1), (input, c_dt)), S_bar).T
            G_paths[step_inx+1] = torch.squeeze(gen_pred)

    return G_paths.cpu().detach().numpy()

if __name__ == '__main__':

    # Loading parameters
    def load_config(filepath, config_key):
        with open(filepath, 'r') as file:
            configs = yaml.safe_load(file)
        return configs[config_key]

    # Load the specific configuration
    config_key = 'config_5'
    config = load_config('parameters.yaml', config_key)

    # Access the variables
    config_name = config['config_name']
    process = config['process']
    S_0 = config['S_0']
    if process == 'GBM' :
        SDE_params = {'mu' : config['mu'], 'sigma' : config['sigma']}
    elif process == 'CIR' :
        SDE_params = {'kappa' : config['kappa'], 'S_bar' : config['S_bar'], 'gamma' : config['gamma']}
    T = config['T']
    use_Z = config['use_Z']

    paths = 5 # dont' do two paths because of line 19/20 (the normalization)

    dt = 0.1
    if process == 'GBM':
        S_E, S_M, Exact_solution, Z, Returns = gen_paths_GBM(S_0, SDE_params['mu'], SDE_params['sigma'], dt, int(T/dt) + 1, paths)
        gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
        gen_model.eval() 
        if use_Z:       
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, None, dt, int(T/dt) + 1, paths, actual_returns=Returns, Z_BM=Z)
        else:
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, None, dt, int(T/dt) + 1, paths, actual_returns=Returns)
    elif process == 'CIR' : 
        S_E, S_M, Exact_solution, Z, Returns = gen_paths_CIR(S_0, SDE_params['kappa'], SDE_params['S_bar'], SDE_params['gamma'], dt, int(T/dt) + 1, paths)
        gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
        gen_model.eval()  
        if use_Z:         
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, SDE_params['S_bar'], dt, int(T/dt) + 1, paths, actual_returns=Returns, Z_BM=Z)
        else:
            model_paths_one_step = gen_paths_from_GAN(gen_model, process, S_0, SDE_params['S_bar'], dt, int(T/dt) + 1, paths, actual_returns=Returns)


    # plt.plot(S_E, color = 'lightblue')
    # plt.plot([], [], color = 'lightblue', label = "Euler")
    # plt.plot(S_M, linestyle='dashed', color = 'palevioletred')
    # plt.plot([], [], color = 'palevioletred', label = "Milstain")
    # plt.plot(Exact_solution, color = "black", linestyle='dashed')
    # plt.plot([], [], color = 'black', label = "Exact")
    # plt.xlabel("time")
    # plt.legend()
    # plt.show()

    model_paths_one_step = np.concatenate([model_paths_one_step[0:1], model_paths_one_step[2:]])
    plt.plot(Exact_solution[:-1], color = 'lightblue')
    plt.plot([], [], color = 'lightblue', label = "Exact")
    plt.plot(model_paths_one_step, linestyle='dashed', color = 'palevioletred')
    plt.plot([], [], color = 'palevioletred', label = "Generated one-step")
    plt.title(f"Generated vs actual paths {config_name} with dt {dt}")
    plt.xlabel("time")
    plt.savefig(f"Plots/Generated vs actual paths_{config_name}")
    plt.legend()
    plt.show()


