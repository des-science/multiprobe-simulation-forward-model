## Notes to understand the script `SP_maps_MagLim.py` and actions required to be able to run it 
### (this page is a work in progress as well as the script that we are understanding how to run from input to output)


#### Goal:

We want to imprint the contamination inferred from the fits of the SP maps to the data to the Maglim and the Metacal galaxy density maps. We start here with MagLim, to recover the existing results.

N.B: The contamination is applied to the galaxy number field, not to the density field itself, since the systematic weights are defined the number of galaxies per pixel.

#### How to make the script running:

Required packages:
- `lsssys`, from https://github.com/des-science/lss_sys/tree/945ebe911abb33b8cb1a5a8159084ee2f9158f3a
- `healpix_util`



**Required files:**

-  the path to `ell.txt` needs to be changed in --> /lss_sys/mock_config/theory_dvecs/cosmosis_y3like/test_theory_maglim_y3like/galaxy_cl/
- need to download` y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits.gz` from https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/redmagic/ . I have currently put it in `/global/cscratch1/sd/vajani/cosmo_packages/lss_sys/files_needed_to_run`


**Folders and files we need to find:** 

- `maglim_v2.2_new_zbinning_jointmask/number_densities.txt`
- `maglim_v2.2_new_zbinning_jointmask/jointmask/tests_post_unblind/enet_weights/w_maps/`


_Notes:_

I have found a folder called `/global/cfs/cdirs/des/monroy/systematics_analysis/enet_weights_tests/w_maps/` but the names of the files are different.

I have asked further information to Martìn and continue digging into `/global/cfs/cdirs/des/` to look for the missing files.




