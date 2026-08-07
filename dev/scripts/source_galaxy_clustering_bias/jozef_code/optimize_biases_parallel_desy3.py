import matplotlib.pyplot as plt
import h5py
import numpy as np
from time import time
from sys import argv, path, exit
from multiprocessing import Pool, cpu_count, set_start_method
import multiprocessing
import os
import pickle as pkl
from tqdm import tqdm



path.append('/global/homes/j/jbucko/desy3/mock_observations/utils/utils_desy3_mocks')
from functions import *
from msfm.utils.input_output import read_yaml
from msfm.utils import files, catalog
# from msfm.utils.power_spectra import *

filename_target_catalog = '/pscratch/sd/j/jbucko/test_catalog_buzzard0_v14.pkl'
catalog_columns = ['ra','dec','e_1','e_2','weight_derived']

do_check_results = True
n_proc_opt = 128

store_path = '/pscratch/sd/j/jbucko/DESY3/v16/desy3_data/source_clustering_biases/'
conf_dir = "/global/homes/j/jbucko/multiprobe-simulation-forward-model/configs/v16/default.yaml"
conf = read_yaml(conf_dir)

n_side = conf["analysis"]["n_side"]
n_pix = conf["analysis"]["n_pix"]

# Metacal tomo bin
tomo_range = range(1, 5)

# load the rotated mask
conf = files.load_config(conf)
# _, patches_pix_dict, _, _ = files.load_pixel_file(conf)
# patch_pix = patches_pix_dict['metacal'][0][0]
# replaced the above by this from v16 on
mask = files.get_mask(conf, nest_out=False)
patch_pix = np.arange(n_pix)[mask]

anti_patch_pix = np.setdiff1d(np.arange(n_pix), patch_pix)

# load DES Y3 data
des_gamma_map, des_count_map = catalog.build_metacal_map_from_cat(conf)
n_z = des_count_map.shape[-1]
des_count_fp = des_count_map[mask,:]
des_count_map[anti_patch_pix] = 0
print(f'des_count_fp shape: {des_count_fp.shape}, {des_count_map.shape}')

# set properties of the 1pt function bins of source clustering counts
range_1pt = (0.5,200.5)
bins_1pt = 200

filename_bias = store_path + 'optimal_biases_desy3.pkl'
filename_loss = store_path + 'optimal_losses_desy3.pkl'
filename_1pt = store_path + 'optimal_1tps_desy3.pkl'

def check_results():
    print(filename_bias)
    with open(filename_bias,'rb') as f:
        optimal_biases = pkl.load(f)
    with open(filename_1pt,'rb') as f:
        optimal_1pt = pkl.load(f)
    
    fig,ax = plt.subplots(1,4,figsize = (20,8),sharey = True)
    
    xvals = np.arange(1,bins_1pt+1)
    for k,v in optimal_1pt.items():
        # print('k,v:',k,v)
        for i in range(4):
            data = v[i][0]
            ax[i].plot(xvals,data,alpha = 0.5, lw = 0.5,color = 'grey')
    for i in range(4):
        data = target_function
        ax[i].plot(xvals,data[i][0],alpha = 1, lw = 1.5,color = 'black')
    fig.supxlabel('# galaxies per pixel')
    fig.supylabel('pixel counts')
    fig.suptitle('metacal')
    # fig.savefig(store_path + 'counts_source_clustering.jpeg',dpi=350)
    fig.savefig('counts_source_clustering.jpeg',dpi=350)

    
    biases = []
    for k,v in optimal_biases.items():
        biases.append(v)
    biases = np.array(biases)
    fig,ax = plt.subplots(1,1,figsize = (10,8))
    minb = np.min(biases)
    maxb = np.max(biases)
    ax.hist(biases[:,0],label = 'b1',bins = 100, range = (minb,maxb),histtype='step')
    ax.hist(biases[:,1],label = 'b2',bins = 100, range = (minb,maxb),histtype='step')
    ax.hist(biases[:,2],label = 'b3',bins = 100, range = (minb,maxb),histtype='step')
    ax.hist(biases[:,3],label = 'b4',bins = 100, range = (minb,maxb),histtype='step')
    ax.legend()
    ax.set_xlabel('biases')
    fig.savefig('biases_source_clustering.jpeg',dpi=350)
    return 0




def load_cosmogrid_metadata(grid = True):
    data_desy3 = h5py.File('/global/cfs/cdirs/des/cosmogrid/CosmoGridV1_metainfo.h5','r')
    if grid == True:
        cosmo_arr = data_desy3['parameters/grid'][()]
    else:
        cosmo_arr = data_desy3['parameters/fiducial'][()]

    grid_cosmo_paths = [item.astype('str').split('/')[-2] for item in cosmo_arr['path_par']]
    cosmo_params_cosmogridv11 = np.c_[cosmo_arr['Om'].astype('float'),cosmo_arr['s8'].astype('float'),cosmo_arr['w0'].astype('float'),cosmo_arr['wa'].astype('float'),cosmo_arr['Ob'].astype('float'),cosmo_arr['ns'].astype('float'),cosmo_arr['H0'].astype('float')/100]
    
    return grid_cosmo_paths,cosmo_params_cosmogridv11

def build_target_funtion(kind = '1pt', des_counts = None):
    """
    prepares the target function from the mock/data to optimize against
    params:
        - kind (string): choose '1pt' for source clustering count histograms, '2pt' for noise power spectra
    return:
        - target_function (list): target function with length 4 for 4 tomo-bins
        - target_n_bar (list): length 4 - number of galaxies per pixel in tomo-bins
    """
    # with open(filename_target_catalog,'rb') as f:
    #     binned_metacal_final_store = pkl.load(f)
        
    # counts_buzz,_ = get_count_map(binned_metacal_final_store,catalog_columns,n_side)
    counts_buzz = des_counts
    print(f"counts Buzzard: {counts_buzz.shape}")
    print(f"same format of DES Y3 data needed!!")
    # exit()
    target_n_bar = np.sum(counts_buzz,axis = 0)/len(patch_pix)
    print('n_bar:',target_n_bar)

    if kind == '1pt':
        h1 = np.histogram(counts_buzz[:,0], bins=bins_1pt,range=range_1pt)
        h2 = np.histogram(counts_buzz[:,1], bins=bins_1pt,range=range_1pt)
        h3 = np.histogram(counts_buzz[:,2], bins=bins_1pt,range=range_1pt)
        h4 = np.histogram(counts_buzz[:,3], bins=bins_1pt,range=range_1pt)
        target_function = [h1,h2,h3,h4]
 
    
    elif kind == '2pt':
        lmax = [589, 863, 1159, 1382]
        cl1 = hp.anafast(counts_buzz[:,0])[30:lmax[0]]
        cl2 = hp.anafast(counts_buzz[:,1])[30:lmax[1]]
        cl3 = hp.anafast(counts_buzz[:,2])[30:lmax[2]]
        cl4 = hp.anafast(counts_buzz[:,3])[30:lmax[3]]
        target_function = [[cl1],[cl2],[cl3],[cl4]]
    else:
        print('kind  %s not implemented, exiting'%kind)
        exit()
    print(f'target function: {target_function}')    
    return target_function,target_n_bar

def get_optimal_biases(i_proc,grid_cosmo_paths,target_function,target_n_bar,optimal_biases_loc,optimal_loss_loc,optimal_stat_loc, kind = '1pt'):
    # get the halo centres first
    for i_grid,grid_cosmo_path in enumerate(grid_cosmo_paths):
        t_start = time()
        biases = []
        losses = []
        grid_dm = get_cosmogrid_normalized_dm_contrast(grid_cosmo_path, perm = 0, n_side = n_side)
        if kind == '1pt':
            for tomo_bin in range(1,5):
                result = minimize_cost_function(get_cost_function, grid_dm, anti_patch_pix, target_n_bar, target_function, tomo_bin = tomo_bin, plot_profile = False)
                # print("Optimal b:", result.x[0])
                # print("Minimum cost function value:", result.fun)
                biases.append(result.x[0])
                losses.append(result.fun)
            
            optimal_counts = get_noisy_counts(grid_dm,biases,target_n_bar,anti_patch_pix,tomo_range)
            h1 = np.histogram(optimal_counts[:,0], bins=bins_1pt,range=range_1pt)
            h2 = np.histogram(optimal_counts[:,1], bins=bins_1pt,range=range_1pt)
            h3 = np.histogram(optimal_counts[:,2], bins=bins_1pt,range=range_1pt)
            h4 = np.histogram(optimal_counts[:,3], bins=bins_1pt,range=range_1pt)
        elif kind == '2pt':
            for tomo_bin in range(1,5):
                result = minimize_cost_function(get_cost_function_2pt, grid_dm, anti_patch_pix, target_n_bar, target_function, tomo_bin = tomo_bin, plot_profile = False)
                # print("Optimal b:", result.x[0])
                # print("Minimum cost function value:", result.fun)
                biases.append(result.x[0])
                losses.append(result.fun)
            
            optimal_counts = get_noisy_counts(grid_dm,biases,target_n_bar,anti_patch_pix,tomo_range)
            lmax = [589, 863, 1159, 1382]
            cl1 = hp.anafast(optimal_counts[:,0])[30:lmax[0]]
            cl2 = hp.anafast(optimal_counts[:,1])[30:lmax[1]]
            cl3 = hp.anafast(optimal_counts[:,2])[30:lmax[2]]
            cl4 = hp.anafast(optimal_counts[:,3])[30:lmax[3]]
            h1,h2,h3,h4 = cl1,cl2,cl3,cl4
        
        optimal_biases_loc[grid_cosmo_path] = biases
        optimal_loss_loc[grid_cosmo_path] = losses
        optimal_stat_loc[grid_cosmo_path] = [[h1],[h2],[h3],[h4]]
        
        print('process %d optimized %d/%d grids'%(i_proc,i_grid+1,len(grid_cosmo_paths)), 'found:',biases,losses, ' time:',time() - t_start)
    return 0
   

if __name__ == "__main__":
    print('available cpus:',cpu_count())
    print('load cosmogrid cosmologies')
    grid_cosmo_paths,grid_cosmo_params = load_cosmogrid_metadata()
    target_function,target_n_bar = build_target_funtion(des_counts = des_count_map)
    if do_check_results == True:
        check_results()
        exit()
    # For SLURM job, you'd want to make sure you have sufficient resources
    manager = multiprocessing.Manager()
    optimal_biases = manager.dict() # will be a list of length n_halos, with halo_id and its centre as elements
    optimal_loss = manager.dict() # will be a list of length n_halos, with halo_id and its centre as elements
    optimal_1pt = manager.dict() # will be a list of length n_halos, with halo_id and its centre as elements
    
    t_global = time()
    for i in range(n_proc_opt):
            globals()['p'+str(i)] = multiprocessing.Process(target=get_optimal_biases, args=(i,grid_cosmo_paths[i::n_proc_opt],target_function,target_n_bar,optimal_biases,optimal_loss,optimal_1pt))
            print('starting process ',i,eval('p'+str(i)))
            eval('p'+str(i)).start()

    for i in range(n_proc_opt):
        print('joining process:',i)
        eval('p'+str(i)).join()
        print(eval('p'+str(i)))

    print('return dict:',len(optimal_biases.keys()),time() - t_global)

    with open(filename_bias,'wb') as f:
        pkl.dump(dict(optimal_biases), f)
    with open(filename_loss,'wb') as f:
        pkl.dump(dict(optimal_loss), f)
    with open(filename_1pt,'wb') as f:
        pkl.dump(dict(optimal_1pt), f)
 