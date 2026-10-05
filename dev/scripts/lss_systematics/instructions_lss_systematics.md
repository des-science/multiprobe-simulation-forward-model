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





__________________________________________________
## TO DOs to apply script to Metacal and generate weights that then we can use to correct the $\delta_g$ cataogue for the sources and see if we get a galaxy bias that makes sense for our noise model

**1st test with the contaminated mocks**
(with ENets which are independant from the mocks and are applied to the mocks as weights)

- Final goal: $\delta_g^{\rm sources}$ --> $w*$ $\delta_g^{\rm sources}$ in the sims - compare histograms
- We need to produce the weights $w$ for the 4 Metacalibration redshift bins, so we will have 4 $w$ maps in the end 
- number_densities.txt files per redshift bins of metacal
- We need the theoretical $C_{\ell}$ with $n(z)$ from the metacal sample. We might have this already from PyCosmo --> organise same format
- Run one set of CosmoGrid with same cosmology as Martin

![Screenshot from 2023-03-29 11-54-13](https://user-images.githubusercontent.com/45969628/228550818-2d72d5f8-b1a7-41d8-8822-85ca2d7678cf.png)


- Do a first run with our $C_{\ell}$  for uncontaminated mocks probably just need to comment	mockrun.add_func('add_weight', mockrun.option_add_weight_maps, options=(weight_maps,)). These uncontaminated mocks we can use for ISDs. 
- We need the mask --> shouldn't be a problem, we can probably can use the same
- to lognormalise the output mocks we need a $k_0$ skewness parameters that for Y3 was found by fitting the distribution of the catalogue with log-normal distribution dependent on $k_0$ (and very dependant on the nside - so Sujeong Lee when doing this for Y3 found a tuning between resolution and reasonable values for $k_0$) - Reference about log-normalization (equation 21): https://arxiv.org/abs/1602.08503 --> Martin contacts Sujeong 
- *Galaxy Bias*: this might be a bit tricky: the code requires the galaxy bias for the sample which is exactly our unknown. It enters the code as `k0    = k0*gbias`. We set up a first test in which we consider that the difference between the catalog's histogram and the fiducial sims histogram is given by an _effective_ bias that get contribution from both the standard galaxy bias $b_{g}$ as well as a bias comeing from the systematics that are not yet corrected for $b_{sys}$ so that $b_{effective}$ = $b_{g}$ * $b_{sys}$. We assume for the first run a $b_{g} = 1$ to have an untuition of the contribution of the systematics and find a way to estimate the actual $b_{g}$ a posteriori through the histogram comparison. 
- We need to produce: 4 E-nets `.fits.gz` files for each redshift bin. *Strategy*: there is a scikit-learn E-nets implementation. What we need to do is then adapt the implementation https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV to our case with the parameters we need (e.g. FRACDET ...) --> Martin will check with Noah


*Description of the output*
```
truncating gaussian ell at theory.lmax-50
Get Xi...
Go back to Cls...
Get Xi...
Go back to Cls...
Get Xi...
Go back to Cls...
Get Xi...
Go back to Cls...
```
what is happening at this step is that mock.lognormalise: this converts your gaussian field into a log-normal field according to equation 3 of arxiv1405.2666 (https://arxiv.org/pdf/1405.2666.pdf)
theory.lognormalise: This converts the input theory C_LN(l) that you want your lognormal fields to have, into the C_G(l) for the gaussian fields. I took this function from FLASK arxiv1602.08503.
The basic idea is (1) generate C_LN(l) with the cosmology/bias you want, convert it into C_G(l), (2) generate gaussian fields with the C_G(l) power spectrum, (3) 'log-normalise' your gaussian fields into log-normal ones using mock.lognoamalise. The measured power spectrum (or w(theta)) of your log-normal fields should now match the input C_LN(l).
