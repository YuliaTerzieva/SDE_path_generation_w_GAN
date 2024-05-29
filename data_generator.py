import numpy as np
import matplotlib.pyplot as plt
import torch


def gen_paths_GBM(S_0 = 1, mu = 0.05, sigma = 0.2, n_steps = 100_000, n_paths = 1000, T = 5_000):  
    # with those default params, dt is 0.05
    # note : in my implementation I don't define delta t, but i define T and n_steps
    
    dt = T/n_steps
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

def gen_paths_from_GAN(gen_model, S_0, dt, n_steps, n_paths, actual = None, my_device = 'mps'): 
    # note the currentl implementation is for iterative generation, 
    # where the next step is generated from the previously generated step
    # that's why only S_0 is provided

    # todo: Add Z parameter for when we have supervised GAN (that uses the Z from the Brownian motion)

    Z = torch.randn((n_steps, n_paths)).type(torch.FloatTensor).to(device=torch.device(my_device))
    # print("Z is", Z.size())
    c_dt = torch.full((1, n_paths), dt).type(torch.FloatTensor).to(device=torch.device(my_device))
    G_paths = torch.zeros((n_steps, n_paths)).type(torch.FloatTensor).to(device=torch.device(my_device))
    # print("G is", G_paths.size())
    G_paths[0, :] = S_0
    for step_inx in range(n_steps-1):
        if actual == None :
            # print("Step", step_inx)
            # print(G_paths[step_inx])
            # print(gen_model(Z[step_inx].view(1, -1), (G_paths[step_inx].view(1, -1), c_dt)))
            # print(torch.mul(G_paths[step_inx].view(-1, 1), gen_model(Z[step_inx].view(1, -1), (G_paths[step_inx].view(1, -1), c_dt))))
            G_paths[step_inx+1] = torch.squeeze(torch.mul(G_paths[step_inx].view(-1, 1), torch.exp(gen_model(Z[step_inx].view(1, -1), (G_paths[step_inx].view(1, -1), c_dt)))))
        else :
            # this should be tested and fixed
            #G_paths[step_inx+1] = G_paths[step_inx] * gen_model(Z[step_inx], (actual[step_inx], c_dt)) 
            pass

    return G_paths.cpu().detach().numpy()

if __name__ == '__main__':
    # np.random.seed(42)
    S_0 = 1
    mu = 0.5
    sigma = 0.2
    n_steps = 100
    paths = 1000
    T = 5
    dt = 0.05
    S_E, S_M, Exact_solution, Z, Log_Return = gen_paths_GBM(S_0, mu, sigma, n_steps, paths, T)
    plt.plot(Exact_solution[:, :10])
    plt.title("Exact solution")
    plt.show()

    plt.plot(Log_Return[:, :10])
    plt.title("Log_Return")
    plt.show()

    gen_model = torch.load('generator_first.pth')
    gen_model.eval()
    
    model_paths = gen_paths_from_GAN(gen_model, S_0, dt, n_steps, 2)

    plt.plot(model_paths)
    plt.title("Generated paths")
    plt.show()