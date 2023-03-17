## Notes to understand the script `SP_maps_MagLim.py` and actions required to run it 
### (this page is a work in progress as well as the script that we are understanding how to run from input to output)


### Goal:

We want to imprint the contamination inferred from the fits of the SP maps to the data to the Maglim and the Metacal galaxy density maps. We start here with MagLim, to recover the existing results.

N.B: The contamination is applied to the galaxy number field, not to the density field itself, since the systematic weights are defined the number of galaxies per pixel.

### How to make the script running:

**Required packages:**
- [lsssys](https://github.com/des-science/lss_sys/tree/945ebe911abb33b8cb1a5a8159084ee2f9158f3a)
- [healpix_util](https://github.com/esheldon/healpix_util)


**Required files:**

-  the path to `ell.txt` needs to be changed in --> `/lss_sys/mock_config/theory_dvecs/cosmosis_y3like/test_theory_maglim_y3like/galaxy_cl/`
- need to download` y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits.gz` from https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/redmagic/ . I have currently put it in `/global/cscratch1/sd/vajani/cosmo_packages/lss_sys/files_needed_to_run`
- `number_densities.txt`
- `maglim_v2.2_new_zbinning_jointmask/jointmask/tests_post_unblind/enet_weights/w_maps/`


**Current state and next step:** 
At the moment through this script we are able to produce Gaussian/log-normal mocks to produce the weights for SP maps for Maglim.
Next step is to repeat the proceudre for Metacal. Since the weightss are very sample specific, we need to re-run any decontamination method on metacal to obtain its own weights. 

Options: 

- run ISD weights: to do this we need to generate lognormal mocks that match metacal's number densities and a skewness parameter for the lognormalization. In addition, we would need the n(z)'s and a given cosmology (we could use Y3 I guess) to generate the Cl's.

- run ENet weights: we just need `N_gal` pixel maps of metacal per redshift bin. N.B.: this is available only at nside = 512. 

- Discuss validation strategy: if we want to validate the weights obtained by any of these methods with lognormal mocks, then we will need to imprint them (1/weights) on the mocks and ideally these weights would come from the method alternative to the one we wish to validate, but this could be discussed to evaluate to what extent we want the validation to go.