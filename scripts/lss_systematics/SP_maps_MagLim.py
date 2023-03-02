"""
Script provided by Martín Rodríguez Monroy to produce Gaussian/log-normal mocks and save them as healpix maps.

This will NOT apply the rescaling.

We want to first apply it to MagLim to recover the results from https://arxiv.org/abs/2105.13540 and once we 
are familiar with it we want to try to apply it to the density maps for Metacal to build the noise model for weak lensing.

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np 
import matplotlib.pyplot as plt 
import lsssys
import healpy as hp 
import healpix_util as hu
import os
import scipy.special
import sys
import fitsio as fio
import time
import errno


def test_dir(value):
        if not os.path.exists(value):
                try:
                        os.makedirs(value, 0o700)
                except OSError as e:
                        if e.errno != errno.EEXIST:
                                raise

try:

    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
	size = comm.Get_size()

except ImportError:

	comm = None
	rank = 0
	size = 1



def main():
	threshold = 2
	if 'test' in sys.argv:
		test = True
		print('{0} unit test mode'.format(sys.argv[0]))

	else:
		test = False

	if test == True:
		label = 'unit_test_output'
		label_extra = ''
		nmocks = 2
		pos_binlist = [1]
		clobber = True
		pos_cldir = 'test_cosmosis_output/galaxy_cl/'

	else:

		label = 'mag_lim_lens_sample_combined_jointmask_newzbinning_contaminated_enet_weights'.format(threshold)
		label_extra = '' #for the output directory
		nmocks = 400
		pos_binlist = 'all'
		clobber = False
		pos_cldir = '/global/cscratch1/sd/vajani/cosmo_packages/lss_sys/mock_config/theory_dvecs/cosmosis_y3like/test_theory_maglim_y3like/galaxy_cl/'
		#pos_cldir = '/scratch/monroy/cosmosis/cosmosis/maglim_v2.2_newzbinning_jointmask/maglim_v2.2_newzbinning_jointmask_test_cosmosis_output/galaxy_cl/'
	nside = 512
	gbias = np.array([1.5,1.8,1.8,1.9,2.3,2.3])
	k0    = np.array([0.584029787721, 0.609685343474, 0.645513012354, 0.637814360022, 0.61694802459, 0.557400091542])
	k0    = k0*gbias



	if rank == 0:

		start = time.time()
		t = lsssys.theory(pos_cldir = pos_cldir, lmax = 3500, pos_binlist=pos_binlist)
		t.lognormalise(k0=k0)
		print('lognormal C(l) took {0}s'.format(start - time.time()))

	else:

		t = None

	if comm is not None:

		t    = comm.bcast(t, root = 0)

	maskfile      = '/global/cscratch1/sd/vajani/cosmo_packages/lss_sys/files_needed_to_run/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits.gz'
	num_dens_file = '/pool/pcae75/data3/des/Y3A2/maglim_sample/maglim_v2.2_new_zbinning_jointmask/number_densities.txt'

	wmaps_path = '/pool/pcae75/data3/des/Y3A2/maglim_sample/maglim_v2.2_new_zbinning_jointmask/jointmask/tests_post_unblind/enet_weights/w_maps/'
	weight_map_files = [
		wmaps_path+'enet_weights_map_maglim_Nbase50_512_izbin0_minfrac0.2_sqrt_pca_hii99_mask_fracdet.fits.gz',
		wmaps_path+'enet_weights_map_maglim_Nbase50_512_izbin1_minfrac0.2_sqrt_pca_hii99_mask_fracdet.fits.gz',
		wmaps_path+'enet_weights_map_maglim_Nbase50_512_izbin2_minfrac0.2_sqrt_pca_hii99_mask_fracdet.fits.gz',
		wmaps_path+'enet_weights_map_maglim_Nbase50_512_izbin3_minfrac0.2_sqrt_pca_hii99_mask_fracdet.fits.gz',
		wmaps_path+'enet_weights_map_maglim_Nbase50_512_izbin4_minfrac0.2_sqrt_pca_hii99_mask_fracdet.fits.gz',
		wmaps_path+'enet_weights_map_maglim_Nbase50_512_izbin5_minfrac0.2_sqrt_pca_hii99_mask_fracdet.fits.gz',
		]

	mock_outdir = '/pool/pcae75/data2/des/Y3A2/maglim_sample/maglim_v2.2_new_zbinning_jointmask/jointmask/tests_post_unblind/enet_weights/weights_validation/contaminated_lognormal_mocks_enet_weights_50_cut/'
	measure_outdir = os.path.join(mock_outdir,'measure/')

	####################################################

	if rank == 0:

		#mock_outdir = os.path.expanduser(mock_outdir)
		test_dir(mock_outdir)
		#if os.path.exists(mock_outdir) == False:

		#	os.mkdir(mock_outdir)
		test_dir(measure_outdir)
		#measure_outdir = os.path.expanduser(measure_outdir)
		#if os.path.exists(measure_outdir) == False:

		#	os.mkdir(measure_outdir)

		mask = lsssys.mask(maskfile, ZMAXcol = None,maskpixname='HPIX',fracpixname='FRACGOOD')
		mask.degrade(nside=nside,minfrac = 0.0)

		weight_maps = []

		for ibin in xrange(t.pos_nbins):

			weight_map_hp_table = fio.read(weight_map_files[ibin])
			weight_map_hp = np.ones(hp.nside2npix(512))*hp.UNSEEN #############################
			weight_map_hp[weight_map_hp_table['HPIX']] = weight_map_hp_table['VALUE']
			if (ibin == 0) and ((weight_map_hp==hp.UNSEEN) == mask.mask).all() == False:

				#apply any additional masks in the weight map to the mask object

				mask.cutmask(cut_array=(weight_map_hp[mask.maskpix] != hp.UNSEEN)  )

				#save the new mask 
				mask.save(mock_outdir+'lognormal_mask.fits.gz', colnames = ['PIXEL','I'], clobber=True)



			weight_map = lsssys.Map()
			weight_map.adddata(weight_map_hp,mask.mask,mask.fracdet)
			#weight_map.degrade(nside, weightedmean=True) ##############################
			weight_maps.append(weight_map.data)

		#mask.degrade(nside=nside,minfrac = 0.0)


	else:

		mock_outdir = None
		measure_outdir = None
		mask = None
		weight_maps = None

	#mask = lsssys.Mask(None,empty=True)
	#mask.maskpix = np.arange(hp.nside2npix(nside))
	#mask.fracpix = np.ones(hp.nside2npix(nside))
	#mask.nside = 512
	#mask.makemask()

	if comm is not None:

		mock_outdir    = comm.bcast(mock_outdir, root = 0)
		measure_outdir = comm.bcast(measure_outdir, root = 0)
		mask           = comm.bcast(mask, root = 0)
		weight_maps    = comm.bcast(weight_maps, root = 0)

	num_dens = np.loadtxt(num_dens_file)

	mockrun = lsssys.mocks.MockRun(t, nside, num_dens, nmocks=nmocks, mask=mask, 
		comm=comm,)

	mockrun.clear_pipeline()
	mockrun.add_func('setup', mockrun.setup)
	mockrun.add_func('lognormalise', mockrun.option_lognormalise, options=(k0,))
	mockrun.add_func('gen_ngal', mockrun.gen_ngal, options=(None,))
	mockrun.add_func('mask', mockrun.option_mask, options=(True,))
	mockrun.add_func('add_weight', mockrun.option_add_weight_maps, options=(weight_maps,))
	mockrun.add_func('poisson', mockrun.option_poisson)
	mockrun.add_func('save', mockrun.save, options=(mock_outdir, label+'_nside{0}'.format(nside), True, 3))
	mockrun.process()

	if rank == 0:
		np.save(measure_outdir+'mockrun_output.npy', mockrun.output)

if __name__ == '__main__':
	main()