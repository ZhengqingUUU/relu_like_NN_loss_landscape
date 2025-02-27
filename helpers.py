import torch
import numpy as np
from model import *
from helpers import *
from tqdm import tqdm
import torch.optim as optim 
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import os 
import shutil
from matplotlib.ticker import ScalarFormatter

FIGSIZE=(6,4)

# Set x-axis labels in scientific notation format
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))  # Adjust the limits as needed

formatter1 = ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((-3, 3))  # Adjust the limits as needed


def generate_5pt_dataset(plot = False, dataset_name= '2D_input_1D_output', plot_name = 'dataset.png'):
    xs = torch.FloatTensor((-1, -0.6, -0.1, 0.3, 0.7)).reshape(5,1)
    x_bias = torch.ones(xs.shape)
    x_data = torch.cat((xs, x_bias), dim=1)
    y_data = torch.FloatTensor((0.28, -0.1, 0.03, 0.23, -0.22)).reshape(5,1)
    if plot:
        os.makedirs(dataset_name, exist_ok=True)
        plt.scatter(xs, y_data, marker='*', c='r')
        plt.savefig(os.path.join(dataset_name, plot_name))
        plt.show()
        plt.clf()
    return x_data, y_data, dataset_name

def generate_2D_dataset(dataset_name= '2D_input_2D_output',):
    x_data = torch.FloatTensor(((-0.3,0.5), (1,1), (-0.6,-1),(0.4,-0.4), ))
    y_data = torch.FloatTensor(( (0.6,-0.5), (0.5,-1),(-0.4,0.6),(0.8,0.2)))
    return x_data, y_data, dataset_name


def generate_3D_input_dataset(dataset_name= '3D_input_1D_output',):
    x_data = torch.FloatTensor(((-0.3,-0.75,-0.5),(-0.2,-0.2,0.4),(-0.6,1,-1),\
                                (-0.4,0.4,0.3), (0.6,-0.1,-0.7),(0.4,-0.9,0.3),\
                                (0.2,0.2,-0.5),))
    y_data = torch.FloatTensor(( -0.5,0.1,-0.6,0.3,0.8,-0.3,-0.1)).reshape(-1,1)
    return x_data, y_data, dataset_name

def cal_betas(mu, mu_p, width):
    """Just a way of parameterization"""
    beta_1 = width**mu
    beta_2 = width**mu_p
    return beta_1, beta_2

def handle_directory(savefolder, config, loss_pic_savedir, loss_savedir, weight_savedir,\
                     error_savedir,  model_savedir, loss_file_name, ):
    if os.path.exists(os.path.join(savefolder, config)):
        shutil.rmtree(os.path.join(savefolder, config)) 
    loss_pic_savedir = os.path.join(savefolder, config, loss_pic_savedir)
    loss_savedir = os.path.join(savefolder, config, loss_savedir)
    weight_savedir = os.path.join(savefolder, config, weight_savedir)
    error_savedir = os.path.join(savefolder, config, error_savedir)
    model_savedir = os.path.join(savefolder, config, model_savedir)
    os.makedirs(loss_pic_savedir, exist_ok=True)
    os.makedirs(loss_savedir, exist_ok=True)
    os.makedirs(weight_savedir, exist_ok=True)
    os.makedirs(error_savedir, exist_ok=True)
    loss_savepath = os.path.join(loss_savedir, loss_file_name)
    
    return loss_pic_savedir, loss_savedir, weight_savedir, error_savedir, model_savedir, loss_savepath

def plot_loss(iter_ls, loss_ls, loss_display_lim, loss_pic_savedir, loss_pic_name, logscale = False):
    plt.figure(figsize=FIGSIZE)
    plt.plot(iter_ls[1:], loss_ls[1:])
    if not logscale:
        plt.ylim([0, loss_display_lim])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if logscale and len(iter_ls)>1:
        plt.yscale('log')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(os.path.join(loss_pic_savedir, loss_pic_name+'.jpg'),dpi = 300)
    plt.clf()

def train(model, x_data, y_data, lr=1e-3, loss_function = F.mse_loss, stopping_loss = 1e-5, max_iter = 1e6, \
         savefolder = 'results', config = 'test',\
         loss_update_step=100, display_loss = True, loss_display_lim = 0.03, loss_pic_savedir = 'pics',  loss_pic_name = 'loss',\
         loss_print_step = 2000, print_loss = True, loss_logscale = False, loss_savedir = 'loss_recording', loss_file_name = 'loss_recording.txt',
         weight_record_step = 500, record_weight = True, weight_savedir = 'weight_recording', \
         error_recording = True, error_record_step = 2000, error_savedir = "error_recording",\
         model_savedir = 'model_optimizer',
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),):
    """
    Train the initialized neural network, plot the loss curve, save input and output weight tensors, together with the error.
    """
    print(config)
    print("training")

    # handling directory maneuvers
    loss_pic_savedir, loss_savedir, weight_savedir, error_savedir,\
          model_savedir, loss_savepath, = handle_directory(savefolder, config, loss_pic_savedir,\
                                                           loss_savedir, weight_savedir,\
                                                            error_savedir, \
                                                                model_savedir, loss_file_name)

    optimizer = optim.SGD(model.parameters(), lr)
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    model = model.to(device)
    global_iter = 0
    loss = torch.tensor(100).float() ## just for initialization
    loss_ls = []
    iter_ls = []
    with tqdm(total=max_iter, position=0, leave=True) as pbar:
        while loss > stopping_loss and global_iter < max_iter:

            pbar.update()

            ## prepare for loss plot
            if (global_iter % loss_update_step == 0) and display_loss:
                loss_ls.append(loss.detach().cpu().numpy())
                iter_ls.append(global_iter)

            ## print the loss
            if global_iter % loss_print_step == 0 and print_loss:
                loss_savepath = os.path.join(loss_savedir, loss_file_name)
                with open(loss_savepath, 'a+') as f:
                    f.write(f"iter: {global_iter}, loss: {loss: .7f}\n")
            
            ## record the  weight matrix
            if global_iter % weight_record_step == 0 and record_weight:
                input_weight = dict(model.named_modules())['input_layer'].weight.data
                output_weight = dict(model.named_modules())['output_layer'].weight.data
                input_weight_savepath = os.path.join(weight_savedir, f"iter{global_iter}_input.pt")
                torch.save(input_weight, input_weight_savepath)
                output_weight_savepath = os.path.join(weight_savedir, f"iter{global_iter}_output.pt")
                torch.save(output_weight, output_weight_savepath)
            
            ## The training routine
            output = model(x_data)
            error =  output.detach().cpu()- y_data.detach().cpu() # will be used
            if global_iter % error_record_step == 0 and error_recording:
                error_savepath = os.path.join(error_savedir, f"iter{global_iter}_error.pt")
                torch.save(error, error_savepath)

            loss = loss_function(output, y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter += 1
        
        iter_arr = np.array(iter_ls)
        loss_arr = np.array(loss_ls)
        np.savez(os.path.join(loss_savedir, "loss.npz"), iter_arr = iter_arr, loss_arr = loss_arr)
    # save the model
    model_savepath = os.path.join(model_savedir,'model_and_optimizer.pt')
    os.makedirs(model_savedir, exist_ok=True)
    torch.save({
            'epoch': global_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_savepath)

    ## After the end of training, plot loss
    plot_loss(iter_ls=iter_ls, loss_ls=loss_ls, loss_display_lim=loss_display_lim,\
                loss_pic_savedir=loss_pic_savedir, loss_pic_name=loss_pic_name, logscale=loss_logscale)

def to_right_range(angle_tensor):
    """ move from (-pi, pi) to (0, 2pi)"""
    right_angle = (angle_tensor + 2*torch.pi)%(2*torch.pi)
    assert min(right_angle) >= 0
    assert max(right_angle) <= 2*torch.pi
    return right_angle


def get_perturb_params(model, width = 50, not_perturb_dead = True, dead_ids = None ,default_min_param_abs = 1,):
    """get the parameters requried for get_perturbed_model function
    """
    # obtain the min abs of the parameters, so the perturbation extent can be determined
    # and count the number of parameters
    min_param_abs = default_min_param_abs
    for layer in model.parameters():
        if not not_perturb_dead:
            examined_layer = layer 
        else: # choose smallest parameters in alive neurons
            effective_layer = layer.squeeze()#TODO: notice, this will squeeze away all the dimensions with size 1, because for the 1d output case, the output weight is of size [1, width], which is very annoying.
            assert effective_layer.shape[0] == width
            # generate the mask indicate the positions of alive neurons
            alive_ids = [i for i in range(width) if i not in dead_ids]
            examined_layer = effective_layer[alive_ids]
        layer_min = torch.min(torch.abs(examined_layer))
        if layer_min < min_param_abs: min_param_abs = layer_min
    perturb_amplitude = min_param_abs/10
    print(f"perturb_amplitude: {perturb_amplitude}")
    return perturb_amplitude.detach().numpy()

def get_perturbed_model(model, perturb_amplitude,  width = 50, not_perturb_dead = True, dead_ids = None,  Gaussian = False):
    """perturb all the parameters with independently sampled noise from a uniform distribution whose range is
    between negative and positive perturb_amplitude. If Gaussian is True, then the noise is sampled independently 
    from a Gaussian distribution.
    """
    with torch.no_grad():
        noise_ls = []
        for layer in model.parameters():
            layer.squeeze_() # squeeze away the dimensions with size 1, the first dimension for each layer should be width
            for id, parameter in enumerate(layer):
                noise = generate_noise(parameter.shape, perturb_amplitude, Gaussian = Gaussian)
                if not_perturb_dead and id in dead_ids: # the first dimension of layer is always the width
                    noise = torch.zeros(parameter.shape)
                parameter.add_(noise)
                # concatenate noise together:
                noise_ls.append(noise.reshape(-1))
                # noise_ls contains the parameters with this sequence: first: 50 * 2 parameters for the input 
                # weights, both components of the same neuron are next to each other. Then, 50 parameters for the 
                # output weights.   
        noise_arr = torch.cat(noise_ls)
    model.eval()
    return model, noise_arr

def generate_noise(shape_tuple, perturb_amplitude, Gaussian = False):
    if not Gaussian:
        noise = perturb_amplitude*(torch.rand(shape_tuple)*2 - 1)
    else:
        noise = torch.normal(mean = torch.zeros(shape_tuple), std = perturb_amplitude*torch.ones(shape_tuple))
    return noise

def perturbation_at_critical_point(mu, mu_p, width, random_seed, savefolder, dataset_generation_func,\
    output_size= 1, perturb_times = 5000, bins = 50, histogram_y_max = 200, Gaussian = False,max_perturb_amplitude = 1,\
    not_perturb_dead = True, dead_ids = None, explanation=False):
    """ This function will perturb the parameters stored in a checkpoint and track the changes in loss 
    given the dataset. The results will be visualized in a histogram, and a dataframe containing the 
    perturbations leading to the negative loss changes will be returned

    Args:
        dataset_generation_func: the function generate the training data points for computing the loss
        bins (int, optional): histogram bin numbers. Defaults to 50.
        histogram_y_max (int, optional): histogram y max. Defaults to 200.

    Returns: a dataframe containing perturbation noises and resulting negative paths.
    Warning: This piece of code uses a lot of bandaid measures to make sure that the perturbation do not change the activation 
    pattern of the neurons so that we only explore the "local" loss landscape. The bandaid measures that I take might not apply to 
    the general case. Please be cautious when reusing this function for other purposes.
    """
    config = f"{mu:.3f}_{mu_p:.3f}_w{width}_rs{random_seed}"
    model_savepath =  os.path.join(savefolder,config,'model_optimizer','model_and_optimizer.pt')
    checkpoint = torch.load(model_savepath)

    net = two_layer_net(2, width, output_size=output_size, beta_1=0, beta_2=0,) # beta_1 and beta_2 are set to zero cuz it does not matter here
    net.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    print(f"loss = {loss}")
    net.eval()
    x_data,y_data,dataset_name = dataset_generation_func()
    

    print("perturbing")
    perturb_df = pd.DataFrame()
    perturb_amplitude = np.min((max_perturb_amplitude,get_perturb_params(net, width = width, not_perturb_dead = not_perturb_dead, dead_ids = dead_ids)))
    print("perturb amplitude: ", perturb_amplitude)
    for i in tqdm(range(perturb_times)):
        perturbed_net, noise = get_perturbed_model(net,perturb_amplitude, width = width,\
            not_perturb_dead = not_perturb_dead, dead_ids=dead_ids ,Gaussian = Gaussian)
        output = perturbed_net(x_data)
        perturbed_loss = F.mse_loss(output.reshape(y_data.shape), y_data)
        loss_change = perturbed_loss - loss
        one_perturb_df = pd.DataFrame({'noise':[noise], 'loss_change': loss_change.detach().numpy()})
        perturb_df = pd.concat([perturb_df,one_perturb_df],ignore_index=True)

    perturb_df.hist(column = 'loss_change',bins=bins)
    loss_change_min = perturb_df['loss_change'].min()
    negative_perturb_df = perturb_df[perturb_df['loss_change'] < 0]
    negative_perturb_df_savedir = os.path.join(savefolder, config, 'perturb_retrain')
    os.makedirs(negative_perturb_df_savedir, exist_ok=True)
    if loss_change_min < 0:
        negative_perturb_df.to_pickle(os.path.join(negative_perturb_df_savedir,'negative_perturb_df.pkl'))
    negative_escape_path_number = (negative_perturb_df.shape)[0]
    if explanation:
        plt.figtext(0.1, -0.1,f"y_max is set to {histogram_y_max}"
                            "\n" f"minimal loss change is {loss_change_min}"
                            "\n"f"total perturbation attempt of {perturb_times} times"
                            "\n" f"acquired {negative_escape_path_number} path(s) to go down the loss surface")
    print(f'min loss change: {loss_change_min}')
    histogram_savepath = os.path.join(savefolder, config,'pics','loss change VS perturbation.jpg')
    plt.grid(False)
    plt.title('')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel('loss changes')
    plt.ylabel('occurences')
    plt.savefig(histogram_savepath, dpi = 300,bbox_inches = 'tight')
    return negative_perturb_df

    
def compute_angle_between_two_angles(angle1, angle2):
    """calculate the angle between two angles. The angle is chosen to be the smaller one between abs(angle1-angle2)
     and 2pi - abs(angle1-angle2)
    """
    angle_subtraction1 = torch.abs(angle1 - angle2).reshape(-1,1)
    angle_subtraction2 = (2*torch.pi - torch.abs(angle1 - angle2)).reshape(-1,1)
    angle_subtraction12 = torch.cat((angle_subtraction1, angle_subtraction2), dim = 1)
    angle_subtraction,_ = torch.min(angle_subtraction12, dim = 1)
    return angle_subtraction

def vertical_line(axs, critical_step_ls, textsize = 10, height=45/44):
    for xc in critical_step_ls:
        axs.axvline(x=xc, color='k', linestyle=':')
        # Calculate vertical position as 45/44 of the y-axis range
        ylim_min, ylim_max = axs.get_ylim()
        y_position = ylim_min + (ylim_max - ylim_min) * height
        
        axs.text(xc-10000, y_position, 
                    f"epoch\n {int(xc/1000)}k", 
                    size=textsize, 
                    verticalalignment='bottom', rotation=90)
    

    