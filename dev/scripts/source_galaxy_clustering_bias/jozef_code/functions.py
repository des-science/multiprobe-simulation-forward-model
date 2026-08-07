import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from msfm.utils import observation, power_spectra,files
from tqdm import tqdm
import h5py,os,sys
import pickle as pkl
from scipy.stats import wasserstein_distance
from copy import deepcopy

base_dir = "/global/homes/j/jbucko/"

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def plot_nz(sim_class,nz_class,normalization = 'ngal', red_factor = 1.0):
    z = sim_class.z_int_centres
    if normalization == 'one':
        plt.plot(z,sim_class.nz1_norm_one(z), label = r'n(z) %s'%sim_class.sim_name,color = 'green',lw = 2.5)
        plt.plot(z,sim_class.nz2_norm_one(z),color = 'green',lw = 2.5)
        plt.plot(z,sim_class.nz3_norm_one(z),color = 'green',lw = 2.5)
        plt.plot(z,sim_class.nz4_norm_one(z),color = 'green',lw = 2.5)
    if normalization == 'ngal':
        plt.plot(z,sim_class.nz1_norm_ng(z), label = r'n(z) sim',color = 'green',lw = 2.5)
        plt.plot(z,sim_class.nz2_norm_ng(z),color = 'green',lw = 2.5)
        plt.plot(z,sim_class.nz3_norm_ng(z),color = 'green',lw = 2.5)
        plt.plot(z,sim_class.nz4_norm_ng(z),color = 'green',lw = 2.5)
    
    plt.plot(z,1/red_factor*nz_class.cg_bin1(z),'k--', label = r'n(z) CosmoGrid')
    plt.plot(z,1/red_factor*nz_class.cg_bin2(z),'k--')
    plt.plot(z,1/red_factor*nz_class.cg_bin3(z),'k--')
    plt.plot(z,1/red_factor*nz_class.cg_bin4(z),'k--')
    
    plt.plot(z,1/red_factor*nz_class.porr_bin1(z),'r--', label = r'n(z) Porredon et al.2021')
    plt.plot(z,1/red_factor*nz_class.porr_bin2(z),'r--')
    plt.plot(z,1/red_factor*nz_class.porr_bin3(z),'r--')
    plt.plot(z,1/red_factor*nz_class.porr_bin4(z),'r--')

    plt.plot(z,1/red_factor*nz_class.cc_bin1(z),'b--', label = r'n(z) C-C')
    plt.plot(z,1/red_factor*nz_class.cc_bin2(z),'b--')
    plt.plot(z,1/red_factor*nz_class.cc_bin3(z),'b--')
    plt.plot(z,1/red_factor*nz_class.cc_bin4(z),'b--')
    
    plt.legend()

def shift_and_stretch(delta_zs, sigma_zs, nz_fid):
    zmeans = np.array([0.296,0.461,0.623,0.776])
    nz_shift_and_stretch = [[]]*4
    nz_shift_and_stretch[0] = lambda z: 1.0/sigma_zs[0]*nz_fid[0]((z - zmeans[0] - delta_zs[0])/sigma_zs[0] + zmeans[0])
    nz_shift_and_stretch[1] = lambda z: 1.0/sigma_zs[1]*nz_fid[1]((z - zmeans[1] - delta_zs[1])/sigma_zs[1] + zmeans[1])
    nz_shift_and_stretch[2] = lambda z: 1.0/sigma_zs[2]*nz_fid[2]((z - zmeans[2] - delta_zs[2])/sigma_zs[2] + zmeans[2])
    nz_shift_and_stretch[3] = lambda z: 1.0/sigma_zs[3]*nz_fid[3]((z - zmeans[3] - delta_zs[3])/sigma_zs[3] + zmeans[3])
    return nz_shift_and_stretch

def flux2mag_des(array):
    """
    convert DES fluxes to magnitudes
    """
    # transform positive fluxes based on DES conversion and assign mag. of 1e6 to negative fluxes
    return np.where(array>0,30 - 2.5*np.log10(array),1e6)

def fill_maglim_bins(mock_observation_class, target_Nz: list):

    N_zs_target_bin1 = target_Nz[0]
    N_zs_target_bin2 = target_Nz[1]
    N_zs_target_bin3 = target_Nz[2]
    N_zs_target_bin4 = target_Nz[3]
    N_zs_target = N_zs_target_bin1+N_zs_target_bin2+N_zs_target_bin3+N_zs_target_bin4

    z = mock_observation_class.z_int_centres
    dz = z[1] - z[0]

    plt.plot(z,N_zs_target,label = r'target distribution DES Y3 maglim (clust. corrected)')

    # bin default maglim cut from sim
    Nz_bufer1 = np.zeros(mock_observation_class.N)
    Nz_bufer2 = np.zeros(mock_observation_class.N)

    mask = (mock_observation_class.catalog[:,mock_observation_class.z_true_idx]<1.5)
    mock_observation_class.catalog = mock_observation_class.catalog[mask,:]
    bins1 = np.digitize(mock_observation_class.catalog[:,mock_observation_class.z_true_idx],mock_observation_class.z_int) - 1
    bins2 = np.digitize(mock_observation_class.catalog_buffer[:,mock_observation_class.z_true_idx],mock_observation_class.z_int) - 1
    unique1, counts1 = np.unique(bins1, return_counts=True)
    unique2, counts2 = np.unique(bins2, return_counts=True)
    Nz_bufer1[unique1] = counts1
    Nz_bufer2[unique2] = counts2
    Nz_bufer1 = Nz_bufer1.astype('int')
    Nz_bufer2 = Nz_bufer2.astype('int')

    plt.plot(z,Nz_bufer1,  label = r'%s distribution - default maglim cut'%mock_observation_class.sim_name)
    plt.plot(z,Nz_bufer2,  label = r'%s buffer'%mock_observation_class.sim_name)
    plt.plot(z,(Nz_bufer1 + Nz_bufer2),  label = r'%s total'%mock_observation_class.sim_name)
    plt.legend(loc = 4)
    plt.show()

    idx_from_base   = []
    idx_from_buffer = []

    count_truth = 0
    count_selected = 0
    n_missing = 0

    missing_y = np.zeros(mock_observation_class.N)
    for i,zs in enumerate(mock_observation_class.z_int_centres):
        Nt  = N_zs_target[i]
        Nt1 = N_zs_target_bin1[i]
        Nt2 = N_zs_target_bin2[i]
        Nt3 = N_zs_target_bin3[i]
        Nt4 = N_zs_target_bin4[i]
        
        Nb1 = Nz_bufer1[i]
        Nb2 = Nz_bufer2[i]

        count_truth += Nt
        if Nb1>Nt:
            # print('case1: i,enough galaxies in buffer 1:',i,Nb1, Nt)
            gal_i = np.where(bins1 == i)[0]
            # print('case1: Nb, vs. gal_i',Nb1,len(gal_i))
            res_idx = np.random.choice(gal_i,size = Nt,replace = False)
            # print('case1: Nt, vs. res_idx',Nt,len(res_idx))
            idx_from_base.append(res_idx)
            count_selected += len(res_idx)
            # np.random.shuffle(res_idx)
            # idx1,idx2,idx3,idx4 = np.array_split(res_idx,[Nt1,Nt1+Nt2,Nt1+Nt2+Nt3])
        elif Nb1 + Nb2 < Nt: 
            # print('case2: i, not enough galaxies in both buffers:',i,Nb1+Nb2, Nt)
            # take all galaxies from both buffers
            gal_i1 = np.where(bins1 == i)[0]
            gal_i2 =  np.where(bins2 == i)[0]
            idx_from_base.append(gal_i1)
            idx_from_buffer.append(gal_i2)

            count_selected += len(gal_i1)
            count_selected += len(gal_i2)
            n_missing += Nt - Nb1 - Nb2
            missing_y[i] = Nt - Nb1 - Nb2
        else:
            # take all from buffer1
            gal_i1 =  np.where(bins1 == i)[0]
            idx_from_base.append(gal_i1)
            count_selected += len(gal_i1)
            
            # we need the following number of galaxies from buffer2
            Ntake = Nt - Nb1
            # print('case3: i, taking few from buffer2,needed in total:',i,Ntake, Nt,' at disposal is: ',Nb2)
            # Here we can either randomly select from buffer or take with the smallest mag_i - 18+4*photo_z         
            gal_i2 =  np.where(bins2 == i)[0]
            # print('case3: gal_i:',gal_i2.shape)
            # buffer_galaxies = mock_observation_class.catalog_buffer[gal_i,:]
            diff = mock_observation_class.catalog_buffer[gal_i2,mock_observation_class.mag_i_true_idx] - (18 + 4*mock_observation_class.catalog_buffer[gal_i2,mock_observation_class.z_dnf_mean_idx])
            # buffer_galaxies = buffer_galaxies[np.argsort(diff),:]
            idx_maglim_cut = np.argsort(diff)
            gal_i2_sorted = gal_i2[idx_maglim_cut]
            # print('diff,sorted indices:',diff[idx_maglim_cut][:Ntake][-1],np.max(diff))
            print('case3: appending Ntake {}/{}:'.format(Ntake,len(idx_maglim_cut)))
            idx_from_buffer.append(gal_i2_sorted[:Ntake])
            count_selected += len(gal_i2_sorted[:Ntake])

    idx_from_base = np.concatenate(idx_from_base)       
    idx_from_buffer = np.concatenate(idx_from_buffer)       

    print(idx_from_base.shape)
    print(idx_from_buffer.shape)

    print('# truth:',count_truth)
    print('# selected:',count_selected)
    print('# missing:',n_missing,n_missing+count_selected)

    selected_from_buffer1 = mock_observation_class.catalog[idx_from_base,:]
    selected_from_buffer2 = mock_observation_class.catalog_buffer[idx_from_buffer,:]
    catalog_final = np.r_[selected_from_buffer1,selected_from_buffer2]
    print('final catalog:',catalog_final.shape)

    bins3 = np.digitize(catalog_final[:,mock_observation_class.z_true_idx],mock_observation_class.z_int) - 1

    unique3, counts3 = np.unique(bins3, return_counts=True)
    Nz_bufer3 = np.zeros(mock_observation_class.N)
    Nz_bufer3[unique3] = counts3

    plt.plot(z,N_zs_target,label = r'target distribution DES Y3 maglim (clust. corrected)')
    plt.plot(z,Nz_bufer1,  label = r'%s distribution - default maglim cut'%mock_observation_class.sim_name)
    plt.plot(z,Nz_bufer2,  label = r'%s buffer'%mock_observation_class.sim_name)
    plt.plot(z,(Nz_bufer1 + Nz_bufer2),  label = r'%s total'%mock_observation_class.sim_name)
    plt.plot(z, Nz_bufer3, label = r'%s final'%mock_observation_class.sim_name)
    plt.legend(fontsize = 5)
    plt.show()

    plt.plot(mock_observation_class.z_int_centres,missing_y)
    plt.show()


    bin_1_final = []
    bin_2_final = []
    bin_3_final = []
    bin_4_final = []

    for i,zs in enumerate(mock_observation_class.z_int_centres):
        Nt  = N_zs_target[i]
        Nt1 = N_zs_target_bin1[i]
        Nt2 = N_zs_target_bin2[i]
        Nt3 = N_zs_target_bin3[i]
        Nt4 = N_zs_target_bin4[i]
        
        Nb1 = Nz_bufer1[i]
        Nb2 = Nz_bufer2[i]

        gal_i = np.where(bins3 == i)[0]
        np.random.shuffle(gal_i)
        if Nb1 + Nb2 < Nt:
            factor = (Nb1+Nb2)/Nt
            # print('Nt, per bin:',Nb1,Nb2,Nt,Nt1,Nt2,Nt3,Nt4)
            adjusted = (np.array([Nt1,Nt2,Nt3,Nt4])*factor).astype('int')
            # print('adjusted:',adjusted)
            diff = Nt - Nb1 - Nb2
            # per bin
            diff = (0.25*diff).astype('int')
            Nt1 -= diff
            Nt2 -= diff
            Nt3 -= diff
            Nt4 -= diff
            # print('Nt, diff, reduced picks:',Nt,diff,Nt1,Nt2,Nt3,Nt4,np.sum([Nt1,Nt2,Nt3,Nt4]))
        idx1,idx2,idx3,idx4 = np.array_split(gal_i,[Nt1,Nt1+Nt2,Nt1+Nt2+Nt3])
        bin_1_final.append(idx1)
        bin_2_final.append(idx2)
        bin_3_final.append(idx3)
        bin_4_final.append(idx4)

    bin_1_final = np.concatenate(bin_1_final)
    bin_2_final = np.concatenate(bin_2_final)
    bin_3_final = np.concatenate(bin_3_final)
    bin_4_final = np.concatenate(bin_4_final)

    return catalog_final,[bin_1_final,bin_2_final,bin_3_final,bin_4_final]

def fill_metacal_bins(mock_observation_class, target_Nz: list):

    N_zs_target_bin1 = target_Nz[0]
    N_zs_target_bin2 = target_Nz[1]
    N_zs_target_bin3 = target_Nz[2]
    N_zs_target_bin4 = target_Nz[3]
    N_zs_target = N_zs_target_bin1+N_zs_target_bin2+N_zs_target_bin3+N_zs_target_bin4

    z = mock_observation_class.z_int_centres
    dz = z[1] - z[0]

    plt.plot(z,N_zs_target,label = r'target distribution DES Y3 metacal (clust. corrected)')

    # bin default maglim cut from sim
    Nz_bufer1 = np.zeros(mock_observation_class.N)
    Nz_bufer2 = np.zeros(mock_observation_class.N)

    # mask = (mock_observation_class.catalog[:,mock_observation_class.z_true_idx]<1.5)
    # mock_observation_class.catalog = mock_observation_class.catalog[mask,:]
    bins1 = np.digitize(mock_observation_class.catalog[:,mock_observation_class.z_true_idx],mock_observation_class.z_int) - 1
    bins2 = np.digitize(mock_observation_class.catalog_buffer[:,mock_observation_class.z_true_idx],mock_observation_class.z_int) - 1
    unique1, counts1 = np.unique(bins1, return_counts=True)
    unique2, counts2 = np.unique(bins2, return_counts=True)
    Nz_bufer1[unique1] = counts1
    Nz_bufer2[unique2] = counts2
    Nz_bufer1 = Nz_bufer1.astype('int')
    Nz_bufer2 = Nz_bufer2.astype('int')

    plt.plot(z,Nz_bufer1,  label = r'%s distribution - default metacal cut'%mock_observation_class.sim_name)
    plt.plot(z,Nz_bufer2,  label = r'%s buffer'%mock_observation_class.sim_name)
    plt.plot(z,(Nz_bufer1 + Nz_bufer2),  label = r'%s total'%mock_observation_class.sim_name)
    plt.legend(loc = 4)
    plt.yscale('log')
    plt.show()

    idx_from_base   = []
    idx_from_buffer = []

    count_truth = 0
    count_selected = 0
    n_missing = 0

    missing_y = np.zeros(mock_observation_class.N)
    # exit()
    for i,zs in enumerate(mock_observation_class.z_int_centres):
        
        Nt  = N_zs_target[i]
        Nt1 = N_zs_target_bin1[i]
        Nt2 = N_zs_target_bin2[i]
        Nt3 = N_zs_target_bin3[i]
        Nt4 = N_zs_target_bin4[i]
        
        Nb1 = Nz_bufer1[i]
        Nb2 = Nz_bufer2[i]

        if i == 109:
            print('attention!')
            print('Nt,Nb1,Nb2:',Nt,Nb1,Nb2)
            print('per-bin targets:',Nt1,Nt2,Nt3,Nt4)

        count_truth += Nt
        if Nb1>Nt:
            # print('case1: i,enough galaxies in buffer 1:',i,Nb1, Nt)
            gal_i = np.where(bins1 == i)[0]
            # print('case1: Nb, vs. gal_i',Nb1,len(gal_i))
            res_idx = np.random.choice(gal_i,size = Nt,replace = False)
            # print('case1: Nt, vs. res_idx',Nt,len(res_idx))
            idx_from_base.append(res_idx)
            count_selected += len(res_idx)
            # np.random.shuffle(res_idx)
            # idx1,idx2,idx3,idx4 = np.array_split(res_idx,[Nt1,Nt1+Nt2,Nt1+Nt2+Nt3])
        elif Nb1 + Nb2 < Nt: 
            # print('case2: i, not enough galaxies in both buffers:',i,Nb1+Nb2, Nt)
            # take all galaxies from both buffers
            gal_i1 = np.where(bins1 == i)[0]
            gal_i2 =  np.where(bins2 == i)[0]
            idx_from_base.append(gal_i1)
            idx_from_buffer.append(gal_i2)

            count_selected += len(gal_i1)
            count_selected += len(gal_i2)
            n_missing += Nt - Nb1 - Nb2
            missing_y[i] = Nt - Nb1 - Nb2
        else:
            # take all from buffer1
            gal_i1 =  np.where(bins1 == i)[0]
            idx_from_base.append(gal_i1)
            count_selected += len(gal_i1)
            
            # we need the following number of galaxies from buffer2
            Ntake = Nt - Nb1
            # print('case3: i, taking few from buffer2,needed in total:',i,Ntake, Nt,' at disposal is: ',Nb2)
            # Here we can either randomly select from buffer or take with the smallest mag_i - 18+4*photo_z         
            gal_i2 =  np.where(bins2 == i)[0]
            # print('case3: gal_i:',gal_i2.shape)
            # buffer_galaxies = mock_observation_class.catalog_buffer[gal_i,:]
            diff = np.ones(len(gal_i2))#mock_observation_class.catalog_buffer[gal_i2,mock_observation_class.mag_i_true_idx] - 23
            # buffer_galaxies = buffer_galaxies[np.argsort(diff),:]
            idx_maglim_cut = np.argsort(diff)
            gal_i2_sorted = gal_i2[idx_maglim_cut]
            # print('diff,sorted indices:',diff[idx_maglim_cut][:Ntake][-1],np.max(diff))
            print('case3: appending Ntake {}/{}:'.format(Ntake,len(idx_maglim_cut)))
            idx_from_buffer.append(gal_i2_sorted[:Ntake])
            count_selected += len(gal_i2_sorted[:Ntake])

    idx_from_base = np.concatenate(idx_from_base)       
    idx_from_buffer = np.concatenate(idx_from_buffer)       

    print(idx_from_base.shape)
    print(idx_from_buffer.shape)

    print('# truth:',count_truth)
    print('# selected:',count_selected)
    print('# missing:',n_missing,n_missing+count_selected)

    selected_from_buffer1 = mock_observation_class.catalog[idx_from_base,:]
    selected_from_buffer2 = mock_observation_class.catalog_buffer[idx_from_buffer,:]
    catalog_final = np.r_[selected_from_buffer1,selected_from_buffer2]
    print('final catalog:',catalog_final.shape)

    bins3 = np.digitize(catalog_final[:,mock_observation_class.z_true_idx],mock_observation_class.z_int) - 1

    unique3, counts3 = np.unique(bins3, return_counts=True)
    Nz_bufer3 = np.zeros(mock_observation_class.N)
    Nz_bufer3[unique3] = counts3

    plt.plot(z,N_zs_target,label = r'target distribution DES Y3 maglim (clust. corrected)')
    plt.plot(z,Nz_bufer1,  label = r'%s distribution - default maglim cut'%mock_observation_class.sim_name)
    plt.plot(z,Nz_bufer2,  label = r'%s buffer'%mock_observation_class.sim_name)
    plt.plot(z,(Nz_bufer1 + Nz_bufer2),  label = r'%s total'%mock_observation_class.sim_name)
    plt.plot(z, Nz_bufer3, label = r'%s final'%mock_observation_class.sim_name)
    plt.legend(fontsize = 5)
    plt.show()

    plt.plot(mock_observation_class.z_int_centres,missing_y)
    plt.show()


    bin_1_final = []
    bin_2_final = []
    bin_3_final = []
    bin_4_final = []

    for i,zs in enumerate(mock_observation_class.z_int_centres):
        Nt  = N_zs_target[i]
        Nt1 = N_zs_target_bin1[i]
        Nt2 = N_zs_target_bin2[i]
        Nt3 = N_zs_target_bin3[i]
        Nt4 = N_zs_target_bin4[i]
        
        Nb1 = Nz_bufer1[i]
        Nb2 = Nz_bufer2[i]

        gal_i = np.where(bins3 == i)[0]
        np.random.shuffle(gal_i)
        if Nb1 + Nb2 < Nt:
            factor = (Nb1+Nb2)/Nt
            # print('Nt, per bin:',Nb1,Nb2,Nt,Nt1,Nt2,Nt3,Nt4)
            adjusted = (np.array([Nt1,Nt2,Nt3,Nt4])*factor).astype('int')
            # print('adjusted:',adjusted)
            diff = Nt - Nb1 - Nb2
            # per bin
            diff = (0.25*diff).astype('int')
            Nt1 = max(0,Nt1 - diff)
            Nt2 = max(0,Nt2 - diff)
            Nt3 = max(0,Nt3 - diff)
            Nt4 = max(0,Nt4 - diff)
            # print('Nt, diff, reduced picks:',Nt,diff,Nt1,Nt2,Nt3,Nt4,np.sum([Nt1,Nt2,Nt3,Nt4]))
        idx1,idx2,idx3,idx4 = np.array_split(gal_i,[Nt1,Nt1+Nt2,Nt1+Nt2+Nt3])
        if i == 109:
            print('attention2!')
            print('adding: ',len(gal_i))
            print('split as: ',len(idx1),len(idx2),len(idx3),len(idx4))
            print('Nt, diff, reduced picks:',Nt,diff,Nt1,Nt2,Nt3,Nt4,np.sum([Nt1,Nt2,Nt3,Nt4]))
        bin_1_final.append(idx1)
        bin_2_final.append(idx2)
        bin_3_final.append(idx3)
        bin_4_final.append(idx4)

    bin_1_final = np.concatenate(bin_1_final)
    bin_2_final = np.concatenate(bin_2_final)
    bin_3_final = np.concatenate(bin_3_final)
    bin_4_final = np.concatenate(bin_4_final)

    return catalog_final,[bin_1_final,bin_2_final,bin_3_final,bin_4_final]

# these are the standard rotation matrices

def get_rot_x(ang):
    return np.array([[1.0, 0.0,         0.0],
                     [0.0, np.cos(ang), -np.sin(ang)],
                     [0.0, np.sin(ang), np.cos(ang)]]).T # Inverse because of healpy

def get_rot_y(ang):
    return np.array([[np.cos(ang),  0.0, np.sin(ang)],
                     [0.0,          1.0, 0.0],
                     [-np.sin(ang), 0.0, np.cos(ang)]]).T # Inverse because of healpy

def get_rot_z(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang), np.cos(ang),  0.0],
                     [0.0,         0.0,          1.0]]).T # Inverse because of healpy

def deg2rad(angles):
    return np.pi/180*angles

def rotate_des_footprint(gal_data_DEC,gal_data_RA):
    # vector positions of the galaxies, shape (num_galaxies, 3)
    vec = hp.ang2vec(theta=gal_data_DEC, phi=gal_data_RA)
    
    # these rotation angles were found by trial and error
    
    y_rot = get_rot_y(-0.125)
    # y_rot.shape = (3,3), rot_vec.shape = (n_galaxies, 3)
    rot_vec = np.dot(y_rot, vec.T)
    
    z_rot = get_rot_z(-1.22)
    # z_rot.shape = (3,3), rot_vec.shape = (3, n_galaxies)
    rot_vec = np.dot(z_rot, rot_vec)
    
    # rot_pix = hp.vec2pix(n_side, rot_vec[0], rot_vec[1], rot_vec[2])
    theta,phi = hp.vec2ang(np.c_[rot_vec[0], rot_vec[1], rot_vec[2]])
    return theta,phi

def plot_clustering_maps(binned_maglim,idx_RA,idx_DEC,n_side = 512,sim_name = 'None',position = 'des',store = False,suffix = '',tf_version = -1):
    if position not in ['des','octant']:
        print('footprint position not known, exiting...')
        exit()
    n_pix = hp.nside2npix(n_side)
    #prepare the array for four clustering (number counts) maps shape (npix,nzbins)
    nonsmoothed_counts = np.zeros((n_pix,4))
    smoothed_counts = np.zeros((n_pix,4))
    # alm are computed for the standard l_max = 3 * n_side - 1
    l_min = 30
    l_max = 1535
    alms_arr = []
    for j in range(1,5):
        RA = binned_maglim[j-1][:,idx_RA]
        DEC = 90-binned_maglim[j-1][:,idx_DEC]
        
        gal_data_RA =  deg2rad(RA)
        gal_data_DEC = deg2rad(DEC) # convert from MICE to healpy convention
        
        if position == 'des':
            theta,phi = rotate_des_footprint(gal_data_DEC,gal_data_RA)
        if position == 'octant':
            theta,phi = gal_data_DEC, gal_data_RA
        all_pix = hp.ang2pix(nside=n_side, theta=theta, phi=phi)
        
        # calculate number counts
        number_count_mask = np.zeros(n_pix)
        for i in range(len(gal_data_RA)):
            pix = all_pix[i]
            number_count_mask[pix] += 1
        # nonsmoothed_counts[:,j-1] = hp.reorder(number_count_mask,r2n=True)
        nonsmoothed_counts[:,j-1] = number_count_mask
        
        # plot
        rest = np.delete(np.arange(n_pix),all_pix)
        number_count_mask[rest] = hp.UNSEEN
        
        hp.mollview(number_count_mask, title="Number counts %s MAGLIM: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0))
        hp.graticule()


        # current_map = number_count_mask.copy()
        # current_map = hp.reorder(current_map, n2r=True)
        # alm = hp.map2alm(current_map, pol=False, use_pixel_weights=True)
        # l = hp.Alm.getlm(3 * n_side - 1)[0]
        # alm[l < l_min] = 0.0
        # alm[l > l_max] = 0.0
        # alms_arr.append(alm)
        # full_map = hp.alm2map(alm, nside=n_side, pol=False)
        # full_map = hp.reorder(full_map, r2n=True)
        # full_map[rest] = hp.UNSEEN
        # hp.mollview(full_map, title="Number counts %s MAGLIM: bin %d - smoothened"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0))
        # hp.graticule()
        mask_file = 'extended.yaml' if position == 'des' else 'linear_bias_octant.yaml'
        conf = base_dir + "multiprobe-simulation-forward-model/configs/%s/%s"%(tf_version,mask_file)
        smoothed_maps_gc, _, pix = observation.forward_model_observation_map(gc_count_map = nonsmoothed_counts,with_padding = False,apply_norm = False,conf = conf,nest_in = False)


        print(smoothed_maps_gc.shape,pix.shape,smoothed_counts[pix,:].shape)
        smoothed_counts[pix,:] = smoothed_maps_gc 
        rest = np.delete(np.arange(n_pix),pix)
        smoothed_counts[rest,:] = hp.UNSEEN
        # reorder to RING
        # for j in range(4):
        #     smoothed_counts[:,j] = hp.reorder(smoothed_counts[:,j],n2r=True)

        hp.mollview(smoothed_counts[:,j-1], title="Number counts %s MAGLIM: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0),nest = True)
        hp.graticule()

        if store:
            filename = "Buzzard_flock/DESY3_mock_observation_%s%s.h5"%(sim_name,suffix)
            print('[plot_clustering_maps] storing maps to: ',filename)
            with h5py.File(filename, "w") as f:
                f.create_group("Header")

                gr1 = f.create_group("maglim")
                gr1.create_dataset("galaxy_counts_bin1",(n_pix,))
                gr1.create_dataset("galaxy_counts_bin2",(n_pix,))
                gr1.create_dataset("galaxy_counts_bin3",(n_pix,))
                gr1.create_dataset("galaxy_counts_bin4",(n_pix,))
                # gr1.create_dataset("pixels",(n_pix,))

                f["maglim"]['galaxy_counts_bin1'][:] = nonsmoothed_counts[:,0]
                f["maglim"]['galaxy_counts_bin2'][:] = nonsmoothed_counts[:,1]
                f["maglim"]['galaxy_counts_bin3'][:] = nonsmoothed_counts[:,2]
                f["maglim"]['galaxy_counts_bin4'][:] = nonsmoothed_counts[:,3]

                gr1.attrs['ordering'] = "RING"
                gr1.attrs['nside'] = n_side
                gr1.attrs['simulation'] = sim_name

            filename = "Buzzard_flock/DESY3_mock_observation_%s%s_forward_modelled.h5"%(sim_name,suffix)
            with h5py.File(filename, "w") as f:
                print('[plot_clustering_maps] storing maps to: ',filename)
                f.create_group("Header")

                gr1 = f.create_group("maglim")
                gr1.create_dataset("galaxy_counts_bin1",(n_pix,))
                gr1.create_dataset("galaxy_counts_bin2",(n_pix,))
                gr1.create_dataset("galaxy_counts_bin3",(n_pix,))
                gr1.create_dataset("galaxy_counts_bin4",(n_pix,))
                # gr1.create_dataset("pixels",(n_pix,))

                f["maglim"]['galaxy_counts_bin1'][:] = smoothed_counts[:,0]
                f["maglim"]['galaxy_counts_bin2'][:] = smoothed_counts[:,1]
                f["maglim"]['galaxy_counts_bin3'][:] = smoothed_counts[:,2]
                f["maglim"]['galaxy_counts_bin4'][:] = smoothed_counts[:,3]

                gr1.attrs['ordering'] = "NEST"
                gr1.attrs['nside'] = n_side
                gr1.attrs['simulation'] = sim_name


    return nonsmoothed_counts, smoothed_counts

def plot_shear_maps(binned_metacal,catalog_columns,total_responses_per_bin_gamma1,total_responses_per_bin_gamma2,n_side = 512,sim_name = 'None',position = 'des',store = False,suffix = '',tf_version = -1):
    if position not in ['des','octant']:
        print('footprint position not known, exiting...')
        exit()
    n_pix = hp.nside2npix(n_side)
    idx_RA,idx_DEC = catalog_columns.index('ra'),catalog_columns.index('dec')
    # idx_e1, idx_e2 = catalog_columns.index('e1_sim'),catalog_columns.index('e2_sim')
    idx_e1, idx_e2 = catalog_columns.index('e_1'),catalog_columns.index('e_2')
    idx_weight = catalog_columns.index('weight_derived')
    #prepare the array for four clustering (number counts) maps shape (npix,nzbins)
    raw_gamma1 = np.zeros((n_pix,4))
    raw_gamma2 = np.zeros((n_pix,4))
    forward_modelled_kappa = np.zeros((n_pix,4))
    n_gal = np.zeros((n_pix,4))

    for j in range(1,5):
        RA = binned_metacal[j-1][:,idx_RA]
        DEC = 90-binned_metacal[j-1][:,idx_DEC]
        
        gal_data_RA =  deg2rad(RA)
        gal_data_DEC = deg2rad(DEC) # convert from MICE to healpy convention
        
        if position == 'des':
            theta,phi = rotate_des_footprint(gal_data_DEC,gal_data_RA)
        if position == 'octant':
            theta,phi = gal_data_DEC, gal_data_RA
        all_pix = hp.ang2pix(nside=n_side, theta=theta, phi=phi)
        
        # calculate mean e1,e2
        e1s = binned_metacal[j-1][:,idx_e1]
        e2s = binned_metacal[j-1][:,idx_e2]
        ws = binned_metacal[j-1][:,idx_weight]
        shapes_mean1 = np.zeros(n_pix)
        shapes_mean2 = np.zeros(n_pix)
        weight_sum = np.zeros(n_pix)
        number_count_mask = np.zeros(n_pix)
        # for i in tqdm(range(n_pix)):
        #     shapes_mean1[i] = np.mean(e1s[all_pix==i])
        #     shapes_mean2[i] = np.mean(e2s[all_pix==i])
        for i in tqdm(range(len(gal_data_RA))):
            pix = all_pix[i]
            shapes_mean1[pix] += ws[i]*e1s[i]
            shapes_mean2[pix] += ws[i]*e2s[i]
            number_count_mask[pix] += 1
            weight_sum[pix] += ws[i]
            # shapes_mean1[pix] += e1s[i]
            # shapes_mean2[pix] += e2s[i]
            # number_count_mask[pix] += 1
            # weight_sum[pix] += 1
    
        shapes_mean1[number_count_mask>0] /= weight_sum[number_count_mask>0]
        shapes_mean2[number_count_mask>0] /= weight_sum[number_count_mask>0]
        print('ignoring responses...')
        raw_gamma1[:,j-1] = shapes_mean1#/total_responses_per_bin_gamma1[j-1]
        raw_gamma2[:,j-1] = shapes_mean2#/total_responses_per_bin_gamma2[j-1]
        n_gal[:,j-1] = number_count_mask
        
        # plot
        rest = np.delete(np.arange(n_pix),all_pix)
        raw_gamma1[rest] = hp.UNSEEN
        raw_gamma2[rest] = hp.UNSEEN
        
        hp.mollview(raw_gamma1[:,j-1], title="gamma1 %s METACAL: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0))
        hp.graticule()

        hp.mollview(raw_gamma2[:,j-1], title="gamma2 %s METACAL: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0))
        hp.graticule()

        mask_file = 'extended.yaml' if position == 'des' else 'linear_bias_octant.yaml'
        conf = base_dir + "multiprobe-simulation-forward-model/configs/%s/%s"%(tf_version,mask_file)
        fw_maps_kappa, _, pix = observation.forward_model_observation_map(wl_gamma_map = np.array((raw_gamma1,raw_gamma2)).transpose(1,2,0),with_padding = False,apply_norm = False,conf = conf,nest_in = False)
        print('fw kappa map:',fw_maps_kappa.shape)
        

        rest = np.delete(np.arange(n_pix),pix)
        forward_modelled_kappa[pix,:] = fw_maps_kappa 
        forward_modelled_kappa[rest,:] = hp.UNSEEN

        hp.mollview(forward_modelled_kappa[:,j-1], title="kappa forward modelled %s METACAL: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0),nest = True)
        hp.graticule()
        
        if store:
            filename = "DESY3_mock_observation_%s%s.h5"%(sim_name,suffix)
            with h5py.File(filename, "w") as f:
                f.create_group("Header")

                gr1 = f.create_group("metacal")
                gr1.create_dataset("raw_gamma1_bin1",(n_pix,))
                gr1.create_dataset("raw_gamma1_bin2",(n_pix,))
                gr1.create_dataset("raw_gamma1_bin3",(n_pix,))
                gr1.create_dataset("raw_gamma1_bin4",(n_pix,))

                gr1.create_dataset("raw_gamma2_bin1",(n_pix,))
                gr1.create_dataset("raw_gamma2_bin2",(n_pix,))
                gr1.create_dataset("raw_gamma2_bin3",(n_pix,))
                gr1.create_dataset("raw_gamma2_bin4",(n_pix,))
                
                gr1.create_dataset("kappa_forward_modelled_bin1",(n_pix,))
                gr1.create_dataset("kappa_forward_modelled_bin2",(n_pix,))
                gr1.create_dataset("kappa_forward_modelled_bin3",(n_pix,))
                gr1.create_dataset("kappa_forward_modelled_bin4",(n_pix,))
                
                f["metacal"]['raw_gamma1_bin1'][:] = raw_gamma1[:,0]
                f["metacal"]['raw_gamma1_bin2'][:] = raw_gamma1[:,1]
                f["metacal"]['raw_gamma1_bin3'][:] = raw_gamma1[:,2]
                f["metacal"]['raw_gamma1_bin4'][:] = raw_gamma1[:,3]

                f["metacal"]['raw_gamma2_bin1'][:] = raw_gamma2[:,0]
                f["metacal"]['raw_gamma2_bin2'][:] = raw_gamma2[:,1]
                f["metacal"]['raw_gamma2_bin3'][:] = raw_gamma2[:,2]
                f["metacal"]['raw_gamma2_bin4'][:] = raw_gamma2[:,3]

                f["metacal"]['kappa_forward_modelled_bin1'][:] = forward_modelled_kappa[:,0]
                f["metacal"]['kappa_forward_modelled_bin2'][:] = forward_modelled_kappa[:,1]
                f["metacal"]['kappa_forward_modelled_bin3'][:] = forward_modelled_kappa[:,2]
                f["metacal"]['kappa_forward_modelled_bin4'][:] = forward_modelled_kappa[:,3]

                gr1.attrs['ordering'] = "NEST"
                gr1.attrs['nside'] = n_side
                gr1.attrs['simulation'] = sim_name

    return [raw_gamma1,raw_gamma2], forward_modelled_kappa, n_gal

def plot_neff_sigmae(bin_metacal,catalog_columns,n_side = 512,sim_name = 'None',position = 'des',des_area = -1):

    if position not in ['des','octant']:
        print('footprint position not known, exiting...')
        exit()
    n_pix = hp.nside2npix(n_side)
    idx_RA,idx_DEC = catalog_columns.index('ra'),catalog_columns.index('dec')
    # idx_e1, idx_e2 = catalog_columns.index('e1_sim'),catalog_columns.index('e2_sim')
    idx_e1, idx_e2 = catalog_columns.index('e_1'),catalog_columns.index('e_2')
    idx_weight = catalog_columns.index('weight_derived')
    #prepare the array for four clustering (number counts) maps shape (npix,nzbins)
    neff_map = np.zeros(n_pix)
    sigmae_map = np.zeros(n_pix)

    RA = bin_metacal[:,idx_RA]
    DEC = 90-bin_metacal[:,idx_DEC]
    
    gal_data_RA =  deg2rad(RA)
    gal_data_DEC = deg2rad(DEC) # convert from MICE to healpy convention
    
    if position == 'des':
        theta,phi = rotate_des_footprint(gal_data_DEC,gal_data_RA)
    if position == 'octant':
        theta,phi = gal_data_DEC, gal_data_RA
    all_pix = hp.ang2pix(nside=n_side, theta=theta, phi=phi)
    idx_gal = [[] for i in range(n_pix)]

    for i in tqdm(range(len(gal_data_RA))):
            pix = all_pix[i]
            idx_gal[pix].append(i)
    
    # sys.exit()
    # calculate mean e1,e2
    e1s = bin_metacal[:,idx_e1]
    e2s = bin_metacal[:,idx_e2]
    ws = bin_metacal[:,idx_weight]
    # unique,indices = np.unique(all_pix, return_index=True)
    # print(unique.shape,indices,len(indices))
    # number_count_mask = np.zeros(n_pix)
    pix_area = hp.nside2pixarea(n_side, degrees = True)*3600 # in arcmin^2
    neff_total = 1/des_area*np.sum(ws)**2/(np.sum(ws*ws))
    tmp = 0.5*np.sum((e1s*ws)**2 + (e2s*ws)**2)/np.sum(ws*ws)
    sigmae_map_total = np.sqrt(tmp)
    print('TOTAL n_eff:',neff_total)
    print('TOTAL sigma_e:',sigmae_map_total)
    for i in tqdm(range(n_pix)):
        mask_pix = idx_gal[i]#indices[i]
        if len(mask_pix)>0:
            if i%10000 == 0:
                print('mask_pix:',len(mask_pix))
                print('n raw:',len(mask_pix)/pix_area)
            # if len(mask_pix)>1:
            weights_pix = ws[mask_pix]
            e1s_pix = e1s[mask_pix]
            e2s_pix = e2s[mask_pix]
            neff_map[i] = 1/pix_area*np.sum(weights_pix)**2/(np.sum(weights_pix*weights_pix))
            tmp = 0.5*np.sum((e1s_pix*weights_pix)**2 + (e2s_pix*weights_pix)**2)/np.sum(weights_pix*weights_pix)
            sigmae_map[i] = np.sqrt(tmp)
    
    # plot
    rest = np.delete(np.arange(n_pix),all_pix)
    neff_map[rest] = hp.UNSEEN
    sigmae_map[rest] = hp.UNSEEN
    
    hp.mollview(neff_map, title="neff %s METACAL: full sample "%(sim_name), cbar=True,bgcolor = 'black',rot=(0,0,0))
    hp.graticule()

    hp.mollview(sigmae_map, title="sigma_e %s METACAL: full sample"%(sim_name), cbar=True,bgcolor = 'black',rot=(0,0,0))
    hp.graticule() 

    hp.mollview(neff_map, title="neff %s METACAL: full sample, limits from paper "%(sim_name), cbar=True,bgcolor = 'black',rot=(0,0,0),min = 4.0, max = 8.0)
    hp.graticule()

    hp.mollview(sigmae_map, title="sigma_e %s METACAL: full sample, limits from paper"%(sim_name), cbar=True,bgcolor = 'black',rot=(0,0,0),min = 0.19, max = 0.25)
    hp.graticule() 

    return neff_map,sigmae_map  

def plot_gamma_kappa_sim(bin_metacal,catalog_columns,n_side = 512,sim_name = 'None',position = 'des',des_area = -1, store = True, suffix = '_from_g1g2',tf_version = -1):

    if position not in ['des','octant']:
        print('footprint position not known, exiting...')
        exit()
    n_pix = hp.nside2npix(n_side)
    idx_RA,idx_DEC = catalog_columns.index('ra'),catalog_columns.index('dec')
    idx_g1, idx_g2, idx_kappa = catalog_columns.index('g1_sim'),catalog_columns.index('g2_sim'),catalog_columns.index('kappa_sim')
    #prepare the array for four clustering (number counts) maps shape (npix,nzbins)
    gamma1_map = np.zeros((n_pix,4))
    gamma2_map = np.zeros((n_pix,4))
    kappa_map = np.zeros((n_pix,4))
    kappa_map_fw = np.zeros((n_pix,4))

    for j in range(1,5):
        RA = bin_metacal[j-1][:,idx_RA]
        DEC = 90-bin_metacal[j-1][:,idx_DEC]
        
        gal_data_RA =  deg2rad(RA)
        gal_data_DEC = deg2rad(DEC) # convert from MICE to healpy convention
        
        if position == 'des':
            theta,phi = rotate_des_footprint(gal_data_DEC,gal_data_RA)
        if position == 'octant':
            theta,phi = gal_data_DEC, gal_data_RA
        all_pix = hp.ang2pix(nside=n_side, theta=theta, phi=phi, nest = True)
        idx_gal = [[] for i in range(n_pix)]

        for i in tqdm(range(len(gal_data_RA))):
                pix = all_pix[i]
                idx_gal[pix].append(i)
        
        # sys.exit()
        # calculate mean e1,e2
        g1s = bin_metacal[j-1][:,idx_g1]
        g2s = bin_metacal[j-1][:,idx_g2]
        kappas = bin_metacal[j-1][:,idx_kappa]

        for i in tqdm(range(n_pix)):
            mask_pix = idx_gal[i]#indices[i]
            if len(mask_pix)>0:
                # if i%100000 == 0:
                #     print('mask_pix:',len(mask_pix))
                #     print('g1 :',g1s[mask_pix])
                #     print('kappa :',kappas[mask_pix])
                # if len(mask_pix)>1:
                g1s_pix = g1s[mask_pix]
                g2s_pix = g2s[mask_pix]
                kappas_pix = kappas[mask_pix]
                
                gamma1_map[i,j-1] = np.mean(g1s_pix)
                gamma2_map[i,j-1] = np.mean(g2s_pix)
                kappa_map[i,j-1] = np.mean(kappas_pix)
        
        # plot
        rest = np.delete(np.arange(n_pix),all_pix)
        gamma1_map[rest,j-1] = hp.UNSEEN
        gamma2_map[rest,j-1] = hp.UNSEEN
        kappa_map[rest,j-1] = hp.UNSEEN

        # rotate gamma1,2,kappa to NEST orgering
        # pix_ring = np.arange(n_pix)
        # pix_nest = hp.ring2nest(n_side,pix_ring)
        # gamma1_map[:,j-1] = gamma1_map[pix_nest,j-1]
        # gamma2_map[:,j-1] = gamma2_map[pix_nest,j-1]
        # kappa_map[:,j-1] = kappa_map[pix_nest,j-1]

    
    j = 3
    hp.mollview(gamma1_map[:,j-1], title="gamma1 %s METACAL: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0),nest = True)
    hp.graticule()

    hp.mollview(gamma2_map[:,j-1], title="gamma2 %s METACAL: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0),nest = True)
    hp.graticule() 

    hp.mollview(kappa_map[:,j-1], title="kappa %s METACAL: bin %d"%(sim_name,j), cbar=True,bgcolor = 'black',rot=(0,0,0),nest = True)
    hp.graticule()

    # forward model kappa map consistently with our pipeline
    mask_file = 'extended.yaml' if position == 'des' else 'linear_bias_octant.yaml'
    conf = base_dir + "multiprobe-simulation-forward-model/configs/%s/%s"%(tf_version,mask_file)
    fw_maps_kappa, _, pix = observation.forward_model_observation_map(wl_gamma_map = np.array((gamma1_map,gamma2_map)).transpose(1,2,0),with_padding = False,apply_norm = False,conf = conf,nest_in = True)
    print('fw kappa map:',fw_maps_kappa.shape)
    

    rest = np.delete(np.arange(n_pix),pix)
    kappa_map_fw[pix,:] = fw_maps_kappa 
    kappa_map_fw[rest,:] = hp.UNSEEN

    if store:
        filename = "DESY3_mock_observation_%s%s.h5"%(sim_name,suffix)
        with h5py.File(filename, "w") as f:
            f.create_group("Header")

            gr1 = f.create_group("metacal")
            gr1.create_dataset("raw_gamma1_bin1",(n_pix,))
            gr1.create_dataset("raw_gamma1_bin2",(n_pix,))
            gr1.create_dataset("raw_gamma1_bin3",(n_pix,))
            gr1.create_dataset("raw_gamma1_bin4",(n_pix,))

            gr1.create_dataset("raw_gamma2_bin1",(n_pix,))
            gr1.create_dataset("raw_gamma2_bin2",(n_pix,))
            gr1.create_dataset("raw_gamma2_bin3",(n_pix,))
            gr1.create_dataset("raw_gamma2_bin4",(n_pix,))
            
            gr1.create_dataset("kappa_forward_modelled_bin1",(n_pix,))
            gr1.create_dataset("kappa_forward_modelled_bin2",(n_pix,))
            gr1.create_dataset("kappa_forward_modelled_bin3",(n_pix,))
            gr1.create_dataset("kappa_forward_modelled_bin4",(n_pix,))
            
            f["metacal"]['raw_gamma1_bin1'][:] = gamma1_map[:,0]
            f["metacal"]['raw_gamma1_bin2'][:] = gamma1_map[:,1]
            f["metacal"]['raw_gamma1_bin3'][:] = gamma1_map[:,2]
            f["metacal"]['raw_gamma1_bin4'][:] = gamma1_map[:,3]

            f["metacal"]['raw_gamma2_bin1'][:] = gamma2_map[:,0]
            f["metacal"]['raw_gamma2_bin2'][:] = gamma2_map[:,1]
            f["metacal"]['raw_gamma2_bin3'][:] = gamma2_map[:,2]
            f["metacal"]['raw_gamma2_bin4'][:] = gamma2_map[:,3]

            f["metacal"]['kappa_forward_modelled_bin1'][:] = kappa_map_fw[:,0]
            f["metacal"]['kappa_forward_modelled_bin2'][:] = kappa_map_fw[:,1]
            f["metacal"]['kappa_forward_modelled_bin3'][:] = kappa_map_fw[:,2]
            f["metacal"]['kappa_forward_modelled_bin4'][:] = kappa_map_fw[:,3]

            gr1.attrs['ordering'] = "NEST"
            gr1.attrs['nside'] = n_side
            gr1.attrs['simulation'] = sim_name

    return gamma1_map,gamma2_map,kappa_map,kappa_map_fw
            
def get_cls_from_mocks(full_sky_maps):
    alms_arr = np.array([hp.map2alm(full_sky_maps[:,i], pol=False, use_pixel_weights=True) for i in range(4)]).T
    print(alms_arr.shape)

    cls = power_spectra.get_cls(alms_arr,True)
    print(cls.shape)
    return cls

def mock_dnf(z_true_train,z_dnf_mean_train,z_true_sim,red_factor = 1):
    np.random.seed(42)
    z_dnf_mean_sim = -np.ones(len(z_true_sim))
    z_true_train = z_true_train[::red_factor]
    z_dnf_mean_train = z_dnf_mean_train[::red_factor]

    N = 250
    z_min = np.min(z_true_train)
    z_max = np.max(z_true_train)
    print('zmin, zmax:',z_min,z_max)
    z_grid = np.linspace(z_min,z_max,N+1)
    z_grid_centres = 0.5*(z_grid[:-1] + z_grid[1:])
    dz = z_grid_centres[1] - z_grid_centres[0]
    
    idx_true_train = np.digitize(z_true_train,z_grid) - 1
    idx_true_sim = np.digitize(z_true_sim,z_grid) - 1

    for i in tqdm(range(len(z_grid_centres))):
        # select true z for a given bin and corresponding photo-z's from DNF
        z_true_train_bin = z_true_train[idx_true_train==i]
        z_dnf_mean_train_bin = z_dnf_mean_train[idx_true_train==i]
        # print(z_true_train_bin,z_dnf_mean_train_bin)
        # bin photo-z's in a given true z-bin
        idx_dnf_mean_train_bin = np.digitize(z_dnf_mean_train_bin,z_grid) - 1
        # build number counts (un-normalized PDF of DNF photo-z per z-bin)
        unique, counts = np.unique(idx_dnf_mean_train_bin, return_counts=True)
        # print(unique,counts)
        Nz_photo_bin = np.zeros(N)
        Nz_photo_bin[unique[1:-1]] = counts[1:-1]
        # print(z_dnf_mean_train_bin,Nz_photo_bin)
        # get number of samples from simulation in the current z_true bins
        nsamples = np.where(idx_true_sim==i)[0].shape[0]
        # get sampled distribution of DNF photo-z
        generated_z_dnf_mean_bin = np.random.choice(z_grid_centres, p = Nz_photo_bin/np.sum(Nz_photo_bin), size = nsamples)
        z_dnf_mean_sim[idx_true_sim==i] = generated_z_dnf_mean_bin + dz*np.random.uniform(-0.5,0.5,nsamples)

    return z_dnf_mean_sim


def msfm_masks(probe = '',bin = 0,mask_type = 'des',tf_version = -1,mask_file = ''):

    if mask_type not in ['des','octant']:
        print('footprint position not known, exiting...')
        exit()
    mask_file = 'extended.yaml' if mask_type == 'des' else 'please_check'
    conf = base_dir + "multiprobe-simulation-forward-model/configs/%s/%s"%(tf_version,mask_file)
    conf = files.load_config(conf)

    # pixels
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = files.load_pixel_file(conf)
    if probe == 'clustering':
        print('loading clustering msfm masks, bin %d'%bin)
        patch_pix = patches_pix_dict["maglim"][bin][0] # bottom-left patch
    if probe == 'lensing':
        print('loading lensing msfm masks, bin %d'%bin)
        patch_pix = patches_pix_dict["metacal"][bin][0] # bottom-left patch
    return patch_pix

def get_weight_response_table(sim_name, return_list = None, catalog_columns = None, work_dir = '/global/u1/j/jbucko/desy3/mock_observations/lensing/'):
    """
    load/create the 2D arrays of size vs 
    """
    data_dir = work_dir + sim_name + '/data/'
    if return_list is not None and not os.path.isfile(data_dir + 'bins_snr.txt'):
        
        print("[get_weight_response_table] computing stored tables...")
        idx_snr = catalog_columns.index('snr')
        idx_T = catalog_columns.index('T')
        idx_mcal_psf_T = catalog_columns.index('mcal_psf_T')
        idx_R11 = catalog_columns.index('R11')
        idx_R22 = catalog_columns.index('R22')
        idx_weight = catalog_columns.index('weight')
        # idx_e1 = catalog_columns.index('e1_sim')
        # idx_e2 = catalog_columns.index('e2_sim')
        idx_e1 = catalog_columns.index('e_1')
        idx_e2 = catalog_columns.index('e_2')
        nbin = 20
        
        # define binning scheme for snr and sizes
        # min_snr = np.min(return_list[:,idx_snr])
        # max_snr = np.max(return_list[:,idx_snr])
        # min_size = np.min(return_list[:,idx_T]/return_list[:,idx_mcal_psf_T])
        # max_size = np.max(return_list[:,idx_T]/return_list[:,idx_mcal_psf_T])
        
        # instead of data-informed min-max, use the same as in paper (Sec. 4.3 of https://arxiv.org/pdf/2011.03408)
        nbin = 20
        min_snr = 10 
        max_snr = 300
        min_size = 0.5
        max_size = 5.0

        bins_snr = np.logspace(np.log10(min_snr),np.log10(max_snr),nbin+1)
        bins_size = np.logspace(np.log10(min_size),np.log10(max_size),nbin+1)

        # bin data based on data above
        dig_snr = np.digitize(return_list[:,idx_snr],bins_snr) - 1
        dig_size = np.digitize(return_list[:,idx_T]/return_list[:,idx_mcal_psf_T],bins_size) - 1
        table_size_snr_weigts = -1e5*np.ones((nbin,nbin))
        table_size_snr_R = -1e5*np.ones((nbin,nbin))
        table_size_snr_R11 = -1e5*np.ones((nbin,nbin))
        table_size_snr_R22 = -1e5*np.ones((nbin,nbin))
        table_size_snr_sigmae = -1e5*np.ones((nbin,nbin))
        table_size_snr_e1 = -1e5*np.ones((nbin,nbin))
        table_size_snr_e2 = -1e5*np.ones((nbin,nbin))
        table_size_snr_R_counts = -1e5*np.ones((nbin,nbin))
        sizes = return_list[:,idx_T]/return_list[:,idx_mcal_psf_T]

        for i in tqdm(range(nbin)):
            for j in range(nbin):
                # size_current_bin = sizes[dig_size == i]
                # snr_current_bin = return_list[dig_snr == j,idx_snr]
                mask = (dig_size == i) & (dig_snr == j)
                weight_current_bin = return_list[mask,idx_weight]
                R11_current_bin = return_list[mask,idx_R11]
                R22_current_bin = return_list[mask,idx_R22]
                R_current_bin = 0.5*(R11_current_bin + R22_current_bin)
                e_1_current_bin = return_list[mask,idx_e1]
                e_2_current_bin = return_list[mask,idx_e2]
                sigmae_per_gal = np.sqrt((return_list[mask,idx_e1]**2 + return_list[mask,idx_e2]**2)/2)

                n_gal = np.sum(mask)
            
                # print(weight_current_bin,R_current_bin)
                table_size_snr_weigts[i,j] = np.mean(weight_current_bin)
                table_size_snr_R[i,j] = np.mean(R_current_bin)
                table_size_snr_R11[i,j] = np.mean(R11_current_bin)
                table_size_snr_R22[i,j] = np.mean(R22_current_bin)
                table_size_snr_e1[i,j] = np.mean(e_1_current_bin)
                table_size_snr_e2[i,j] = np.mean(e_2_current_bin)
                table_size_snr_sigmae[i,j] = np.sqrt(np.mean(sigmae_per_gal**2))
                table_size_snr_R_counts[i,j] = n_gal
                print('i=%d,j=%d, w, R, #'%(i,j),table_size_snr_weigts[i,j],table_size_snr_R[i,j],table_size_snr_sigmae[i,j],table_size_snr_R_counts[i,j])

                np.savetxt(data_dir + 'bins_snr.txt',bins_snr)
                np.savetxt(data_dir + 'bins_size.txt',bins_size)
                np.savetxt(data_dir + 'table_size_snr_weigts.txt',table_size_snr_weigts)
                np.savetxt(data_dir + 'table_size_snr_R.txt',table_size_snr_R)
                np.savetxt(data_dir + 'table_size_snr_R11.txt',table_size_snr_R11)
                np.savetxt(data_dir + 'table_size_snr_R22.txt',table_size_snr_R22)
                np.savetxt(data_dir + 'table_size_snr_sigmae.txt',table_size_snr_sigmae)
                np.savetxt(data_dir + 'table_size_snr_e1.txt',table_size_snr_e1)
                np.savetxt(data_dir + 'table_size_snr_e2.txt',table_size_snr_e2)
                np.savetxt(data_dir + 'table_size_snr_R_counts.txt',table_size_snr_R_counts)
    else:
        print("[get_weight_response_table] loading stored tables...")
        bins_snr              = np.genfromtxt(data_dir + 'bins_snr.txt')
        bins_size             = np.genfromtxt(data_dir + 'bins_size.txt')
        table_size_snr_weigts = np.genfromtxt(data_dir + 'table_size_snr_weigts.txt')
        table_size_snr_R      = np.genfromtxt(data_dir + 'table_size_snr_R.txt')
        table_size_snr_R11      = np.genfromtxt(data_dir + 'table_size_snr_R11.txt')
        table_size_snr_R22      = np.genfromtxt(data_dir + 'table_size_snr_R22.txt')
        table_size_snr_sigmae      = np.genfromtxt(data_dir + 'table_size_snr_sigmae.txt')
        table_size_snr_e1          = np.genfromtxt(data_dir + 'table_size_snr_e1.txt')
        table_size_snr_e2          = np.genfromtxt(data_dir + 'table_size_snr_e2.txt')
        table_size_snr_R_counts      = np.genfromtxt(data_dir + 'table_size_snr_R_counts.txt')
        print("[get_weight_response_table] loading stored tables done!")
    
    return bins_size,bins_snr,table_size_snr_weigts,table_size_snr_R,table_size_snr_R11,table_size_snr_R22,table_size_snr_sigmae,table_size_snr_e1,table_size_snr_e2,table_size_snr_R_counts

def get_bin_size_snr(size,snr,bins_size,bins_snr):
    dig_size = np.digitize(size,bins_size) - 1
    dig_snr = np.digitize(snr,bins_snr) - 1
    # print('dig size:',dig_size)
    # if the value == max value, then decrease bin, otherwise error when querying the table
    dig_size[dig_size == len(bins_size) - 1] = len(bins_size) - 2
    dig_snr[dig_snr == len(bins_snr) - 1] = len(bins_snr) - 2

    return dig_size,dig_snr


def get_cosmogrid_normalized_dm_contrast(path,perm = 0,n_side = 512,grid = True):
    """
    load cosmology from path (e.g. closest to buzzard in the Om-sigma8-w0 plane, but not only)
    params:
        path: cosmogrid local path of the simulation
        perm: permutation index
        n_side: n_side of the tomographic density contrast maps
        grid (bool): whether to load from grid or fiducial 
    returns: 
        dg_map: array of shape (n_pix,4) of normalized dark matter density contrast
    """
    np.random.seed(42)
    if grid == True:
        projected_maps_dir = "/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/%s/perm_%04d/projected_probes_maps_v11dmb.h5"%(path,perm)
    else:
        projected_maps_dir = "/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/%s/perm_%04d/projected_probes_maps_v11dmb.h5"%(path,perm)
    dg_maps = []
    
    for i, tomo in enumerate(range(1,5)):
        with h5py.File(projected_maps_dir, "r") as f:
            dg_map = f[f"map/dg/metacal{tomo}"][:]
            dg_map = hp.ud_grade(dg_map,n_side,order_out = 'RING')
    
        # normalize the number density contrast
        dg_map = (dg_map - np.mean(dg_map))/np.mean(dg_map)
        dg_maps.append(dg_map)

    return np.array(dg_maps).T

def get_noisy_counts(dm_maps,metacal_biases,n_bar_tomo,anti_patch_pix,tomo_range = [1,2,3,4]):
    """
    convert normalized dark matter density contrast to noisy galaxy counts
    params:
        dm_maps: array of shape (n_pix,4) of normalized dark matter density contrast
        metacal_biases: source clustering biases to apply
        n_bar_tomo: mean galaxy number counts/pixel for all the bins,
        anti_patch_pix: pixels, where the densoty contrast should be zero, shape (n_pix,)
        tomo_range: list, range of tomographic indices to evaluate the density contrast for
    returns: 
        noisy_sim_counts_tomo: array of shape (n_pix,4) of noisy galaxy number counts
    """
    
    noisy_sim_counts_tomo = []
    # print('tomo range:',tomo_range,metacal_biases)
    for i, tomo in enumerate(tomo_range):
        sim_counts = (1+(metacal_biases[i])*dm_maps[:,tomo-1])*n_bar_tomo[tomo-1]
    
        # dealing with negative values
        ng_clip = np.clip(sim_counts, a_min=0, a_max=None, dtype=np.float32)
        sim_counts = ng_clip * np.sum(sim_counts) / np.sum(ng_clip)

        np.random.seed(42)
        noisy_sim_counts = np.random.poisson(sim_counts)
        # noisy_sim_counts = sim_counts
        noisy_sim_counts = noisy_sim_counts.astype(np.float32)
    
        # apply masking
        noisy_sim_counts[anti_patch_pix] = 0
    
        # sim_counts_tomo.append(sim_counts)
        noisy_sim_counts_tomo.append(noisy_sim_counts)
    return np.array(noisy_sim_counts_tomo).T

from msfm.utils.power_spectra import *
def get_noise_shear_from_counts(noisy_sim_counts_tomo_cosmox, conf):
    """
    get noisy shear maps and count and shear Cls
    params:
        noisy_sim_counts_tomo_cosmox:   tomographic galaxy counts of shape (n_pix,4)
        conf: (files.load_config object)
    returns:
        noisy_sim_counts_tomo_cosmox:   tomographic galaxy counts of shape (n_pix,4) - basically the input
        gamma_noise_cg_cosmox:          tomographic noise shear (gamma0) of shape (n_pix,4)
        obs_cl_cg_noise_cosmox_counts:  source clustering power spectra (n_bins,4), n_bins set to 32
        obs_cl_cg_noise_cosmox_gamma:   shear noise power spectra, length = 2 [gamma1,gamma2] with shape of each of them (n_bins,4), n_bins set to 32
        ell_bins: ell's, shape (n_bins,)
        obs_fm: output of forward model, being: observation, observation_cls, footprint_pix, smoothed_binnned_kappa_cls
    """
    
    l_mins = [0]*4
    l_maxs = [1535]*4
    ell_bins = get_cl_bins(0, l_maxs[0], 32)
    # get shear
    gamma_noise_cg_cosmox = observation.make_shape_noise_map(noisy_sim_counts_tomo_cosmox.astype('int'), conf, noise_seed=12)

    alms_noise_cg_cosmox_counts = get_alms(noisy_sim_counts_tomo_cosmox)
    cls_noise_cg_cosmox_counts = get_cls(alms_noise_cg_cosmox_counts, with_cross = False)

    alms_noise_cg_cosmox_gamma0 = get_alms(gamma_noise_cg_cosmox[...,0])
    cls_noise_cg_cosmox_gamma0 = get_cls(alms_noise_cg_cosmox_gamma0, with_cross = False)
    alms_noise_cg_cosmox_gamma1 = get_alms(gamma_noise_cg_cosmox[...,1])
    cls_noise_cg_cosmox_gamma1 = get_cls(alms_noise_cg_cosmox_gamma1, with_cross = False)
    
    obs_cl_cg_noise_cosmox_counts, _ = power_spectra.smooth_and_bin_cls(
        cls_noise_cg_cosmox_counts,
        l_mins_smoothing=l_mins,
        l_maxs_smoothing=l_maxs,
        with_cross=False,
        fixed_binning=True,
        n_bins=conf["analysis"]["power_spectra"]["n_bins"],
        l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
        l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
    )

    obs_cl_cg_noise_cosmox_gamma0, _ = power_spectra.smooth_and_bin_cls(
        cls_noise_cg_cosmox_gamma0,
        l_mins_smoothing=l_mins,
        l_maxs_smoothing=l_maxs,
        with_cross=False,
        fixed_binning=True,
        n_bins=conf["analysis"]["power_spectra"]["n_bins"],
        l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
        l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
    )

    obs_cl_cg_noise_cosmox_gamma1, _ = power_spectra.smooth_and_bin_cls(
        cls_noise_cg_cosmox_gamma1,
        l_mins_smoothing=l_mins,
        l_maxs_smoothing=l_maxs,
        with_cross=False,
        fixed_binning=True,
        n_bins=conf["analysis"]["power_spectra"]["n_bins"],
        l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
        l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
    )

    obs_kappa_footprint,obs_kappa_cls,obs_pix = observation.forward_model_observation_map(wl_gamma_map = deepcopy(gamma_noise_cg_cosmox),conf = conf,apply_norm = False,with_padding=False,nest_in=False)
    obs_cl_cg_noise_cosmox_kappa, _ = power_spectra.smooth_and_bin_cls(
        obs_kappa_cls[:,[0,4,7,9]],
        l_mins_smoothing=l_mins,
        l_maxs_smoothing=l_maxs,
        with_cross=False,
        fixed_binning=True,
        n_bins=conf["analysis"]["power_spectra"]["n_bins"],
        l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
        l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
    )

    

    return noisy_sim_counts_tomo_cosmox,gamma_noise_cg_cosmox,obs_cl_cg_noise_cosmox_counts,[obs_cl_cg_noise_cosmox_gamma0,obs_cl_cg_noise_cosmox_gamma1],ell_bins,[obs_kappa_footprint,obs_kappa_cls,obs_pix,obs_cl_cg_noise_cosmox_kappa]

def get_wasserstein_bin(counts_bin,histogram_bin_des,edges = (0.5,200.5),nbins = 200):
    # print('counts_bin:',counts_bin.shape)
    h = np.histogram(counts_bin, bins=nbins,range=edges)
    
    # plt.scatter(range(1,201),histogram_bin_des)
    # plt.scatter(range(1,201),h[0])
    # plt.title('optimization scatter')
    # plt.show()
    # print('#objects in des, buzz:',np.sum(histogram_bin_des),np.sum(h[0]))
    return wasserstein_distance(h[0],histogram_bin_des)

def get_log_wasserstein_bin(counts_bin,histogram_bin_des,edges = (0.5,200.5),nbins = 200):
    h = np.histogram(counts_bin, bins=nbins,range=edges)
    return wasserstein_distance(np.log10(h[0]+1e-50),np.log10(histogram_bin_des+1e-50))


def get_cost_function(metacal_biases,dm_maps,n_bar_tomo,anti_patch_pix,histogram_bin_des,tomo_range,log = False):
    if len(tomo_range)!=1:
        print('tomo_range has to be of length 1 (e.g. [0])')
        sys.exit()
    # print('[get_cost_function]tomo_range:',tomo_range)
    noisy_counts = get_noisy_counts(dm_maps,metacal_biases,n_bar_tomo,anti_patch_pix,tomo_range)
    if log == False:
        # print('log=False')
        distance = get_wasserstein_bin(noisy_counts,histogram_bin_des)
    else:
        distance = get_log_wasserstein_bin(noisy_counts,histogram_bin_des)
        # print('log=True')
    # print(distance)
    return distance

def get_cost_function_2pt(metacal_biases,dm_maps,n_bar_tomo,anti_patch_pix,cls_bin_des,tomo_range,log = False):
    if len(tomo_range)!=1:
        print('tomo_range has to be of length 1 (e.g. [0])')
        sys.exit()
    # print('[get_cost_function_2pt]tomo_range:',tomo_range)
    noisy_counts = get_noisy_counts(dm_maps,metacal_biases,n_bar_tomo,anti_patch_pix,tomo_range)
    lmax = [589, 863, 1159, 1382]
    for i in range(noisy_counts.shape[1]):
        # print('bin %d, mean counts: %.2f'%(i+1,np.mean(noisy_counts[:,i])))
        print(lmax[i])
        cl_bin = hp.anafast(noisy_counts[:,i],)[30:lmax[i]]
        # collect the Cls for the current bin
        if i == 0:
            obs_cl_cg_noise_cosmox_counts = cl_bin
        else:
            obs_cl_cg_noise_cosmox_counts = np.vstack((obs_cl_cg_noise_cosmox_counts,cl_bin))

    # print('obs_cl_cg_noise_cosmox_counts shape:',obs_cl_cg_noise_cosmox_counts.shape)
    # print('cls_bin_des shape:',cls_bin_des,cls_bin_des.shape)
    # print('stats:',tomo_range[0])

    ell = np.arange(len(obs_cl_cg_noise_cosmox_counts.flatten()))+30
    distance = np.sum(((np.log10(obs_cl_cg_noise_cosmox_counts.flatten()) - np.log10(cls_bin_des.flatten())))**2)
    
    # print(f"distance:{distance}")

    # ell = np.arange(len(obs_cl_cg_noise_cosmox_counts.flatten()))
    # plt.semilogx(ell, np.log10(obs_cl_cg_noise_cosmox_counts.flatten()))
    # plt.semilogx(ell, np.log10(cls_bin_des.flatten()))
    # plt.title(f'optimization scatter, biases: {metacal_biases[0]:.3f}, bin {tomo_range[0]}, dist:{distance:.3f}')
    # # plt.title(f'optimization scatter, biases: {metacal_biases:f}')
    # plt.show()
    return distance

def get_count_map(binned_metacal,catalog_columns,n_side = 512,position = 'des'):
    """
    get galaxy number count maps from the binned catalog
    params:
        - binned_metacal: binned catalog with length = 4, each element has shape (#gal_in_bin,len(catalog_columns)
        - catalog_columns: columns in the binned catalog, list
        - n_side: n_side of the final maps
        - position: 'des' or 'octant', determines whether to rotate the footprint or not
    returns:
        - n_gal: tomographic counts maps (n_pix,4)
        - seq_pix: list with length = 4, each element is healpy.ang2pix for all the galaxies in a given map
    """
    if position not in ['des','octant']:
        print('footprint position not known, exiting...')
        exit()
    n_pix = hp.nside2npix(n_side)
    idx_RA,idx_DEC = catalog_columns.index('ra'),catalog_columns.index('dec')
    n_gal = np.zeros((n_pix,4))
    seq_pix = []

    for j in range(1,5):
        # angle convention conversion
        RA = binned_metacal[j-1][:,idx_RA]
        DEC = 90-binned_metacal[j-1][:,idx_DEC]
        
        # convert from degrees to radians
        gal_data_RA =  deg2rad(RA)
        gal_data_DEC = deg2rad(DEC) # convert from MICE to healpy convention
        
        if position == 'des':
            theta,phi = rotate_des_footprint(gal_data_DEC,gal_data_RA)
        if position == 'octant':
            theta,phi = gal_data_DEC, gal_data_RA
        
        all_pix = hp.ang2pix(nside=n_side, theta=theta, phi=phi)
        number_count_mask = np.zeros(n_pix)
    
        for i in tqdm(range(len(gal_data_RA))):
            pix = all_pix[i]
            number_count_mask[pix] += 1
            
        n_gal[:,j-1] = number_count_mask
        seq_pix.append(all_pix)

    return n_gal,seq_pix

from scipy.optimize import  minimize
def minimize_cost_function(get_cost_function, dm_maps, anti_patch_pix, n_bar_tomo, des_count_histograms, tomo_bin=1, log = False, plot_profile = True, b_init=1.0):
    """
    Automatically minimizes get_cost_function over the bias parameter `b`.

    Parameters:
    - get_cost_function: Function to be minimized
    - dm_maps: Dark matter maps (input to get_cost_function)
    - anti_patch_pix: (n_pix,4) healpy pixels with 0 galaxies
    - n_bar_tomo: Number density for tomographic bin
    - des_count_histograms: Histogram bin descriptions
    - tomo_bin: Tomographic bin index (default is 1)
    - b_init: Initial guess for `b` (default is 1.0)

    Returns:
    - result: Optimization result from scipy.optimize.minimize
    """

    # Define wrapper function for minimization
    def cost_function_to_minimize(b):
        # print('log1:',log)
        return get_cost_function([b[0]], dm_maps, n_bar_tomo, anti_patch_pix, des_count_histograms[tomo_bin-1][0], [tomo_bin], log)

    # Perform minimization
    result = minimize(cost_function_to_minimize, x0=[b_init], method='Nelder-Mead')

    # if plot_profile == True:
    #     # Plot the function profile
    #     bs = np.linspace(0.1, 5, 20)  # Range of `b` values
    #     profile = [get_cost_function([b], dm_maps, n_bar_tomo, anti_patch_pix, des_count_histograms[tomo_bin-1][0], tomo_range=[tomo_bin]) for b in bs]
    
    #     plt.plot(bs, profile, label="Cost function profile")
    #     plt.axvline(result.x[0], color='r', linestyle='--', label=f"Optimal b = {result.x[0]:.3f}")
    #     plt.xlabel("b")
    #     plt.ylabel("Cost function value")
    #     plt.legend()
    #     plt.show()

    return result


def store_mock(galaxy_counts,raw_gamma1,raw_gamma2,forward_modelled_kappa,sim_name,n_side,suffix,hp_order = 'NEST',store_path = '/pscratch/sd/j/jbucko/DESY3/mock_observations/lensing/'):
    n_pix = hp.nside2npix(n_side)
    filename = store_path+"DESY3_mock_observation_%s%s.h5"%(sim_name,suffix)
    with h5py.File(filename, "w") as f:
        f.create_group("Header")
    
        gr1 = f.create_group("metacal")

        gr1.create_dataset("galaxy_counts_bin1",(n_pix,))
        gr1.create_dataset("galaxy_counts_bin2",(n_pix,))
        gr1.create_dataset("galaxy_counts_bin3",(n_pix,))
        gr1.create_dataset("galaxy_counts_bin4",(n_pix,))

        gr1.create_dataset("raw_gamma1_bin1",(n_pix,))
        gr1.create_dataset("raw_gamma1_bin2",(n_pix,))
        gr1.create_dataset("raw_gamma1_bin3",(n_pix,))
        gr1.create_dataset("raw_gamma1_bin4",(n_pix,))
    
        gr1.create_dataset("raw_gamma2_bin1",(n_pix,))
        gr1.create_dataset("raw_gamma2_bin2",(n_pix,))
        gr1.create_dataset("raw_gamma2_bin3",(n_pix,))
        gr1.create_dataset("raw_gamma2_bin4",(n_pix,))
        
        gr1.create_dataset("kappa_forward_modelled_bin1",(n_pix,))
        gr1.create_dataset("kappa_forward_modelled_bin2",(n_pix,))
        gr1.create_dataset("kappa_forward_modelled_bin3",(n_pix,))
        gr1.create_dataset("kappa_forward_modelled_bin4",(n_pix,))
        
        f["metacal"]['galaxy_counts_bin1'][:] = galaxy_counts[:,0]
        f["metacal"]['galaxy_counts_bin2'][:] = galaxy_counts[:,1]
        f["metacal"]['galaxy_counts_bin3'][:] = galaxy_counts[:,2]
        f["metacal"]['galaxy_counts_bin4'][:] = galaxy_counts[:,3]

        f["metacal"]['raw_gamma1_bin1'][:] = raw_gamma1[:,0]
        f["metacal"]['raw_gamma1_bin2'][:] = raw_gamma1[:,1]
        f["metacal"]['raw_gamma1_bin3'][:] = raw_gamma1[:,2]
        f["metacal"]['raw_gamma1_bin4'][:] = raw_gamma1[:,3]
    
        f["metacal"]['raw_gamma2_bin1'][:] = raw_gamma2[:,0]
        f["metacal"]['raw_gamma2_bin2'][:] = raw_gamma2[:,1]
        f["metacal"]['raw_gamma2_bin3'][:] = raw_gamma2[:,2]
        f["metacal"]['raw_gamma2_bin4'][:] = raw_gamma2[:,3]
    
        f["metacal"]['kappa_forward_modelled_bin1'][:] = forward_modelled_kappa[:,0]
        f["metacal"]['kappa_forward_modelled_bin2'][:] = forward_modelled_kappa[:,1]
        f["metacal"]['kappa_forward_modelled_bin3'][:] = forward_modelled_kappa[:,2]
        f["metacal"]['kappa_forward_modelled_bin4'][:] = forward_modelled_kappa[:,3]
    
        gr1.attrs['ordering'] = hp_order
        gr1.attrs['nside'] = n_side
        gr1.attrs['simulation'] = sim_name

def make_shear_map(wl_counts_map, wl_gamma_catalog_binned, pix_ids_catalog, conf, noise_seed=12, with_noise = True):
    """
    function returning the shear metacal map from WL counts maps. Slightly extended fuction of mfsm.utils.observation.make_shape_noise_map
    params:
        wl_counts_map (2D array): tomographic counts per pixel on a full sky; shape (n_pix,n_tomo_bins)
        wl_gamma_catalog_binned (list or arrays): len = n_tomo_bins, each element returns a binned catalog of shape (n_gal,2) - WL shear at the position of all the galaxies
        pix_ids_catalog (list of arrays): len = n_tomo_bins, each element is an array of healpix pixel id of a given galaxy; shape (n_gal_bin,)
        conf: config file as a JSON/dictionary
        noise_seed: noise seed
    """
    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.random.set_seed(noise_seed)

    # constants
    n_pix = conf["analysis"]["n_pix"]
    _, patches_pix_dict, _, _ = files.load_pixel_file(conf)

    tomo_gamma_cat, _ = files.load_noise_file(conf)

    gamma1 = []
    gamma2 = []
    for i in range(wl_counts_map.shape[-1]):
        patch_pix = patches_pix_dict["metacal"][i][0]
        wl_gamma_catalog = wl_gamma_catalog_binned[i]

        with tf.device("/CPU:0"):
            counts = wl_counts_map[patch_pix, i]

            # create joint distribution, as this is faster than random indexing
            gamma_abs = tf.math.abs(tomo_gamma_cat[i][:, 0] + 1j * tomo_gamma_cat[i][:, 1])
            w = tomo_gamma_cat[i][:, 2]
            cat_dist = tfp.distributions.Empirical(samples=tf.stack([gamma_abs, w], axis=-1), event_ndims=1)

            gamma1_noise, gamma2_noise = shear_gen(counts, patch_pix, wl_gamma_catalog, pix_ids_catalog[i], cat_dist, n_noise_per_example=1, with_noise = with_noise, i_bin = i)
            gamma1_noise = gamma1_noise[:, 0]
            gamma2_noise = gamma2_noise[:, 0]

        gamma1_patch = np.zeros(n_pix, dtype=np.float32)
        gamma1_patch[patch_pix] = gamma1_noise

        gamma2_patch = np.zeros(n_pix, dtype=np.float32)
        gamma2_patch[patch_pix] = gamma2_noise

        gamma1.append(gamma1_patch)
        gamma2.append(gamma2_patch)

    gamma1 = np.stack(gamma1, axis=-1)
    gamma2 = np.stack(gamma2, axis=-1)

    return np.stack([gamma1, gamma2], axis=-1)

def shear_gen(counts, patch_pix, wl_gamma_catalog, pix_ids_catalog, cat_dist, n_noise_per_example, with_noise, i_bin):
    """Generates shape noise and WL shear from a map of galaxy counts and WL catalog and joint distribution of absolute shear values and their
    weights.

    Args:
        counts (np.ndarray): Array of shape (len(base_patch_pix),) that contains the galaxy count per pixel
        patch_pix (np.ndarray): Array of indices that are part of the survey patch
        wl_gamma_catalog (np.ndarray): Array of shape (n_gal_bin,2) containing WL shear at a position of each galaxy in a given bin
        pix_ids_catalog (np.array): healpy indices of all the galaxies in a given bin
        cat_dist (tfp.distributions): Distribution with samples of length 2 that contains the absolute magnitudes and
            weights
        n_noise_per_example (int): Number of noise realizations to create, this dimension is included for vectorization

    Returns:
        np.ndarray: Arrays of shape (len(base_patch_pix, n_noise_per_example) containing the two gamma components
    """

    import tensorflow as tf

    # indices to sum over all of the galaxies in the individual pixels

    seg_ids = []
    wl_gamma1_seg = []
    wl_gamma2_seg = []
    ordering = []
    mock_indices = np.arange(wl_gamma_catalog.shape[0])
    e1e2w_matched = np.zeros((np.sum(counts),5))
    print('counts:',counts.shape,np.sum(counts),'vs.',wl_gamma_catalog.shape[0])
    dict_map = build_dict_arr1_arr2(patch_pix,pix_ids_catalog)
    sanity_count = 0
    print('init state:',wl_gamma_catalog[:10,:])
    for id, n_gals in tqdm(enumerate(counts)):
        seg_ids.extend(n_gals * [id])
        # print('pix_ids_catalog,[patch_pix[id]]:',pix_ids_catalog,id,patch_pix)
        mask = dict_map[patch_pix[id]]
        # if np.sum(mask) != n_gals:
            # print('sum mask, ngal:',np.sum(mask),n_gals)
        wl_gamma1_seg.extend(list(wl_gamma_catalog[mask,0]))
        wl_gamma2_seg.extend(list(wl_gamma_catalog[mask,1]))
        
        
        idx = np.where(mask)[0]
        # ordering.extend(idx)
        ordering.extend(list(mask))
        sanity_count += n_gals
        if len(wl_gamma1_seg)!=len(ordering):
            print('something wrong here: %d/%d'%(id,len(counts)),len(wl_gamma1_seg),len(ordering),n_gals,sanity_count,len(idx),idx)

    print('counts:',sanity_count,len(ordering),np.sum(counts))
    # make a tensor, this is important for performance
    print('seg_ids:',len(seg_ids),len(wl_gamma1_seg))
    seg_ids = tf.constant(seg_ids, dtype=tf.int32)
    wl_gamma1_seg = tf.constant(np.array(wl_gamma1_seg).reshape(-1,1), dtype = tf.float32)
    wl_gamma2_seg = tf.constant(np.array(wl_gamma2_seg).reshape(-1,1), dtype = tf.float32)

    # total number of galaxies in the patch
    n_gals_patch = len(seg_ids)

    # shape (n_gals_patch, n_noise_per_example, 2)
    cat_samples = cat_dist.sample(sample_shape=(n_gals_patch, n_noise_per_example))
    # shape (n_gals_patch, n_noise_per_example)
    phase_samples = tf.random.uniform(
        shape=(
            n_gals_patch,
            n_noise_per_example,
        ),
        minval=0,
        maxval=2 * np.pi,
    )

    # shape (n_gals_patch, n_noise_per_example)
    w_samples = cat_samples[..., 1]

    if with_noise == True:
        g1_samples_noise = tf.math.cos(phase_samples) * cat_samples[..., 0]
        g2_samples_noise = tf.math.sin(phase_samples) * cat_samples[..., 0]
        # print('shape noise, shape WL:',g1_samples_noise.shape,wl_gamma1_seg.shape,type(g1_samples_noise),type(wl_gamma1_seg))
        g1_samples = g1_samples_noise + wl_gamma1_seg # here will also come responses at some point
        g2_samples = g2_samples_noise + wl_gamma2_seg # here will also come responses at some point
        # print('ordering:',ordering[:10],wl_gamma1_seg[:10])
        e1e2w_matched[ordering,:] = np.c_[wl_gamma1_seg,wl_gamma2_seg,g1_samples_noise,g2_samples_noise,w_samples]
        # with open(f'/pscratch/sd/j/jbucko/DESY3/mock_observations/lensing/buzzard_flock/8/e1e2w_i_bin_{i_bin}.pkl','wb') as f:
        #     pkl.dump(e1e2w_matched,f)
        # with open(f'/pscratch/sd/j/jbucko/DESY3/mock_observations/lensing/buzzard_flock/8/ordering_i_bin_{i_bin}.pkl','wb') as f:
        #     pkl.dump(ordering,f)
    else:
        g1_samples =  wl_gamma1_seg # here will also come responses at some point
        g2_samples =  wl_gamma2_seg # here will also come responses at some point
    # shape (n_gals_patch, n_noise_per_example, 3)
    # print('shape:',g1_samples.shape)


    """
    this is sum w_i(e_i + gamma_i) / sum w_i
    """
    # weighted_gamma_samples = tf.stack([g1_samples * w_samples, g2_samples * w_samples, w_samples], axis=-1)
    # sum_per_pix = tf.math.segment_sum(weighted_gamma_samples, seg_ids)
    #  # normalize with weights, set 0/0 equal to 0 instead of nan
    # gamma_per_pix = tf.math.divide_no_nan(sum_per_pix[..., :2], tf.expand_dims(sum_per_pix[..., 2], axis=-1))

    """
    this is sum w_i * e_i / sum w_i + sum gamma_i / N, N is a number of galaxies in a given pixel
    """
    weighted_gamma_samples = tf.stack([(g1_samples - wl_gamma1_seg) * w_samples, (g2_samples - wl_gamma2_seg) * w_samples, w_samples], axis=-1)
    sum_per_pix = tf.math.segment_sum(weighted_gamma_samples, seg_ids)
     # normalize with weights, set 0/0 equal to 0 instead of nan
    gamma_per_pix = tf.math.divide_no_nan(sum_per_pix[..., :2], tf.expand_dims(sum_per_pix[..., 2], axis=-1)) # now only the shape noise part

    # now the gravity shear part
    weights_unit = tf.ones_like(w_samples, dtype=tf.float32)
    weighted_gamma_samples = tf.stack([wl_gamma1_seg * weights_unit, wl_gamma2_seg * weights_unit, weights_unit], axis=-1)
    sum_per_pix = tf.math.segment_sum(weighted_gamma_samples, seg_ids)
     # normalize with weights, set 0/0 equal to 0 instead of nan
    print('[shear_gen] adding mean gravity shear and metacal-weighted shape noise')
    gamma_per_pix += tf.math.divide_no_nan(sum_per_pix[..., :2], tf.expand_dims(sum_per_pix[..., 2], axis=-1)) # now only the shape noise part




    # The condition means that the final pixel contains zero galaxies. Then, its index is not included in the seg_ids
    # (multiplication with zero) and because it's the last, tensorflow has no way of knowing that it should still take
    # the segmented_sum over this index, which evaluates to zero. The while loop allows more than one of the last
    # pixels to be zero.
    n_final_zero_pix = 0
    while counts[-(n_final_zero_pix + 1)] == 0:
        n_final_zero_pix += 1

    if n_final_zero_pix > 0:
        # There is no galaxy in the final pixels, so the shape noise there is equal to zero
        zero_pix = tf.zeros((n_final_zero_pix, n_noise_per_example, 2), dtype=tf.float32)
        gamma_per_pix = tf.concat((gamma_per_pix, zero_pix), axis=0)

    # shape (len(base_patch_pix), n_noise_per_example)
    return gamma_per_pix[..., 0].numpy(), gamma_per_pix[..., 1].numpy()

def build_dict_arr1_arr2(arr1,arr2):
    """
    function returning occurances of elements of arr1 elements in arr2
    params:
        arr1 (list): list of values
        arr2 (list): list of values containing elements of arr1 (with repetition)
    returs:
        result (dict): keys are elements of arr1, values are indices of a given key in arr2
    """
    from collections import defaultdict

    # One pass through arr2 to build a mapping from value to list of indices
    arr2_index_map = defaultdict(list)
    for idx, val in tqdm(enumerate(arr2)):
        arr2_index_map[val].append(idx)

    # Now map each arr1 value to its list of indices in arr2 (if any)
    result = {val: arr2_index_map[val] for val in arr1 if val in arr2_index_map}
    return result

def get_kappa_cls_map_from_shear_maps(shear_maps,conf,with_cross = False):
    l_mins = [0]*4
    l_maxs = [1535]*4
    ell_bins = get_cl_bins(0, l_maxs[0], 32)
    
    obs_kappa_footprint,obs_kappa_cls,obs_pix = observation.forward_model_observation_map(wl_gamma_map = shear_maps.copy(),conf = conf,apply_norm = False,with_padding=False,nest_in=False)

    if with_cross == True:
        cls = obs_kappa_cls[:,:]
    else:
        cls = obs_kappa_cls[:,[0,4,7,9]]
    obs_cl_cg_noise_cosmox_kappa, _ = power_spectra.smooth_and_bin_cls(
        cls,
        l_mins_smoothing=l_mins,
        l_maxs_smoothing=l_maxs,
        with_cross=with_cross,
        fixed_binning=True,
        n_bins=conf["analysis"]["power_spectra"]["n_bins"],
        l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
        l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
    )
    return ell_bins,[obs_kappa_footprint,obs_kappa_cls,obs_pix,obs_cl_cg_noise_cosmox_kappa]

def find_cosmo(cosmo_list,cosmo_target,params):
    cosmo_target_list = []
    for par in params:
        print(f'param: {par}')
        t_value = cosmo_target[par]
        cosmo_target_list.append(t_value)
    indices = []
    for i in range(cosmo_list.shape[0]):
        if np.all(cosmo_list[i,:] == cosmo_target_list): indices.append(i)

    if len(indices) == 0:
        from scipy.spatial import KDTree
        print('trying to return closest cosmology using KDTree')
        cosmo_tree = KDTree(cosmo_list)
        indices = cosmo_tree.query(cosmo_target_list)[1]

    return indices
    