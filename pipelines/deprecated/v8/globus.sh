export GLOBUS_PERLMUTTER="6bdc7956-fc0f-4ad2-989c-7aa5ee643a79"
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/6bdc7956-fc0f-4ad2-989c-7aa5ee643a79/data_access]'
globus ls "$GLOBUS_PERLMUTTER"

export GLOBUS_EULER_SCRATCH="21ab72af-db90-40a5-9b00-f990481caabb"
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/21ab72af-db90-40a5-9b00-f990481caabb/data_access]'
globus ls "$GLOBUS_EULER_SCRATCH"

export GLOBUS_EULER_WORK="41088e99-feb9-41f0-8148-3a89a5c64874"
globus ls "$GLOBUS_EULER_WORK"

globus transfer "$GLOBUS_PERLMUTTER:/global/cfs/cdirs/des/cosmogrid/v11desy3/fiducial/cosmo_fiducial/perm_0000" "$GLOBUS_EULER_SCRATCH:/cluster/scratch/athomsen/debug/globus"
globus transfer "$GLOBUS_PERLMUTTER:/global/cfs/cdirs/des/cosmogrid/v11desy3/fiducial/cosmo_fiducial/perm_0001" "$GLOBUS_EULER_SCRATCH:/cluster/scratch/athomsen/debug/globus"