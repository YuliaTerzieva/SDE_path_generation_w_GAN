from plots import *
from training import *
import yaml

# Loading parameters
def load_config(filepath, config_key):
    with open(filepath, 'r') as file:
        configs = yaml.safe_load(file)
    return configs[config_key]

# Load the specific configuration
config_key = 'config_2'
config = load_config('parameters.yaml', config_key)

# Access the variables
config_name = config['config_name']
process = config['process']
S_0 = config['S_0']
if process == 'GBM' :
    SDE_params = {'mu' : config['mu'], 'sigma' : config['sigma']}
elif process == 'CIR' :
    SDE_params = {'kappa' : config['kappa'], 'S_bar' : config['S_bar'], 'gamma' : config['gamma']}
n_steps = config['n_steps']
n_paths = config['n_paths']
dt = config['dt']
number_data_points = config['number_data_points']
epochs = config['epochs']
batch_size = config['batch_size']
advancing_C = config['advancing_C']
log_freq = config['log_freq']
use_Z = config['use_Z']


# We first train the networks
# D_losses, G_losses = train_network(config_name, process, S_0, SDE_params, n_steps, n_paths, dt, number_data_points, epochs, batch_size, advancing_C, log_freq, use_Z)

steps_weak_stong = np.arange(20, 100, 20)
paths_weak_strong = 500
weak_stong_error_gen_paths(config_name, process, S_0, SDE_params, steps_weak_stong, paths_weak_strong, dt, use_Z)