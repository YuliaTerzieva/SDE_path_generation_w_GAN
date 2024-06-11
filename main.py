from plots import *
from training import *
import yaml

# Loading parameters
def load_config(filepath, config_key):
    with open(filepath, 'r') as file:
        configs = yaml.safe_load(file)
    return configs[config_key]

# Load the specific configuration
config_key = 'config_6'
config = load_config('parameters.yaml', config_key)

# Access the variables
config_name = config['config_name']
process = config['process']
S_0 = config['S_0']
if process == 'GBM' :
    SDE_params = {'mu' : config['mu'], 'sigma' : config['sigma']}
elif process == 'CIR' :
    SDE_params = {'kappa' : config['kappa'], 'S_bar' : config['S_bar'], 'gamma' : config['gamma']}
dts = config['dts']
T = config['T']
use_Z = config['use_Z']
number_data_points = config['number_data_points']
epochs = config['epochs']
batch_size = config['batch_size']
advancing_C = config['advancing_C']
log_freq = config['log_freq']

torch.random.manual_seed(42)
np.random.seed(42)

### We first train the networks ###
# print('\033[35m'+ f"I'm starting training of {config_name} with process {process} w/ parameters {SDE_params} \
#       \n the training is on {number_data_points} points, note using Z is {use_Z}" + '\033[30m')
# D_losses, G_losses = train_network(config_name, process ,S_0, SDE_params, dts, T, use_Z, number_data_points, \
#                                    epochs, batch_size, advancing_C, log_freq)

gen_model = torch.load(f'Trained_Models/generator_{config_name}.pth')
gen_model.eval()

# ECDF_plot(gen_model, config_name, process, S_0, SDE_params, use_Z)

# torch.random.manual_seed(42)
# np.random.seed(42)
# ks_plot(gen_model, config_name, process, S_0, SDE_params, use_Z)

# weak_stong_error(gen_model, config_name, process, S_0, SDE_params, dts, T, batch_size, use_Z)
            
# supervised_vs_not_generator_map()

### Note : run this only if you have a supervised model configuration ###
d_model = torch.load(f'Trained_Models/discriminator_{config_name}.pth')
d_model.eval()
discriminator_map(d_model, config_name, process, S_0)