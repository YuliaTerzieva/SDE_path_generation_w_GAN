This project is a reproducibility study of the paper 
[**_Monte Carlo simulation of SDEs using GANs_** by van Rhijn et al.](https://link.springer.com/article/10.1007/s13160-022-00534-x)


### Project structure

There are two GANs trained for each of the 3 SDEs - one conditional and one supervised. 
The GAN architecture is the same for the conditional and the supervised variants and can be found in [networks.py](https://github.com/YuliaTerzieva/SDE_path_generation_w_GAN/blob/main/networks.py)

The parameters used for those 3 SDEs are organized in configurations in [parameters.yaml](https://github.com/YuliaTerzieva/SDE_path_generation_w_GAN/blob/main/parameters.yaml) 
_Note : initially I trained models on a single time step instead of multiple and those parameters and the plots for those are in parameters_old.yaml and Plots_old_

Everything is called and runs from [main.py](https://github.com/YuliaTerzieva/SDE_path_generation_w_GAN/blob/main/main.py)
If you'd like to add another configuration with different parameters, add that configuration to parameters.yaml and load the configuration in main.py. 
That would take care of the train and producing all the plots

The training function that gets called in the  main can be found in [training.py](https://github.com/YuliaTerzieva/SDE_path_generation_w_GAN/blob/main/training.py)

In that function, the training data is being generated and used. All the data generation including the making of the training data, 
the making of SDE paths and the transformation between $S_t \rightarrow R_t \rightarrow S_t$ is organised in [data_generation.py](https://github.com/YuliaTerzieva/SDE_path_generation_w_GAN/blob/main/data_generator.py). 

All the plots are saved in a folder organized by type of plot. All the names follow the naming convention "Plots\**type of plot**\**type of plot**_gan type_SDE type", 
for example "Plots/ECDF/ECDF_cGAN_CIR_no_Feller.png" is ECDF plot of the conditional GAN with CIR SDE with Feller condition not satisfied. 

All the networks are saved in Trained_Models are can be loaded as shown in the main.py. 

Lastly, the presentation I gave on the project is attached [here](https://github.com/YuliaTerzieva/SDE_path_generation_w_GAN/blob/main/Presentation.pdf)
