import os
import h5py

from msfm.utils import logger, files, parameters

LOGGER = logger.get_logger(__file__)


def _check_metacal_bias_matches_forward_model(conf, survey_systematics):
    """Assert that the source clustering bias table was fit against the forward model that is about to use it.

    The bias is fixed by matching the simulated one-point function of the source counts to DES Y3, so a table fit
    against a clean model has the imaging systematics absorbed into b. Combining it with a forward model that also
    imprints them double counts the systematics, and the reverse leaves them out entirely. Neither shows up as an
    error anywhere downstream, hence this check.

    Args:
        conf (dict): Configuration.
        survey_systematics (bool): Whether the forward model imprints the imaging systematics, see
            files.get_shape_noise.
    """
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))

    # files.metacal_bias_arm selects a group within the file, see files.read_metacal_bias
    arm = conf["files"].get("metacal_bias_arm")
    with h5py.File(os.path.join(repo_dir, conf["files"]["metacal_bias"]), "r") as f:
        root = f[arm] if arm else f
        # tables predating this attribute were all fit against a clean forward model
        table_label = str(root.attrs.get("systematics_label", "none"))

    status = ""
    if survey_systematics:
        with h5py.File(os.path.join(repo_dir, conf["files"]["metacal_systematics"]), "r") as f:
            conf_label = str(f.attrs["label"])
            status = str(f.attrs.get("status", ""))
    else:
        conf_label = "none"

    assert table_label == conf_label, (
        f"The metacal bias table {conf['files']['metacal_bias']} was fit against the imaging systematics "
        f"{table_label!r}, but the forward model applies {conf_label!r}. Point files.metacal_bias at the table that "
        f"was fit with the same shape_noise.survey_systematics setting"
    )
    LOGGER.info(f"Metacal source clustering bias table fit against the imaging systematics {table_label!r}")

    # the label is free text, so it says which lss_sys run was meant, not what the maps contain. The export carries
    # its own caveats and they belong in the run log rather than only in an attribute nobody reads
    if status:
        LOGGER.warning(f"Imaging systematics {conf_label!r} self-reports: {status}")


def check_survey_paths(conf):
    """Assert that every repo path in files: and dirs: belongs to the configured survey: it lies directly in data/
    (the inputs shared by all surveys) or under data/<survey.name in lower case>/. Absolute paths are not checked.

    Args:
        conf (dict): Configuration.
    """
    survey_dir = conf["survey"]["name"].lower()
    for section in ("files", "dirs"):
        for key, path in conf.get(section, {}).items():
            if isinstance(path, str) and path.startswith("data/"):
                parts = path.split("/")
                assert len(parts) == 2 or parts[1] == survey_dir, (
                    f"{section}.{key} = {path} is neither a shared input in data/ nor under data/{survey_dir}/, "
                    f"the directory of survey.name {conf['survey']['name']}"
                )


def print_and_check_modeling_in_config(conf):
    check_survey_paths(conf)

    LOGGER.info("Modeling choices:")
    for key, value in dict(conf["analysis"]["modelling"]).items():
        if isinstance(value, dict):
            LOGGER.info(f"{key}:")
            for k, v in value.items():
                LOGGER.info(f"  {k} = {v}")
        else:
            LOGGER.info(f"{key} = {value}")

    if conf["analysis"]["modelling"]["degrade_to_grf"]:
        LOGGER.warning("Degrading to Gaussian Random Field")

    # lensing
    # every sampled parameter (label) has either a uniform or a Gaussian prior, the latter only on the Latin hypercube
    # since CosmoGrid fixes the Sobol points
    grid = conf["analysis"]["grid"]
    params = conf["analysis"]["params"]
    sobol = params["cosmo"] + (params.get("bary") or []) + (params.get("stochasticity") or [])
    labels = parameters.get_parameters(conf=conf)
    gaussian_priors = grid.get("gaussian_priors") or {}
    for param, prior in gaussian_priors.items():
        assert param in labels and param not in sobol, f"the Gaussian prior on {param} needs a Latin hypercube label"
        assert param not in grid["priors"], f"{param} has both a uniform and a Gaussian prior"
        assert len(prior) == 4, f"the Gaussian prior on {param} must be [mean, standard deviation, lower, upper]"
        mu, sigma, lo, hi = prior
        assert sigma > 0 and lo <= mu <= hi, f"the Gaussian prior on {param} needs sigma > 0 and mu in [{lo}, {hi}]"
        LOGGER.info(f"{param} has the prior N({mu}, {sigma}) truncated to [{lo}, {hi}]")

    # the grid takes an IA parameter by name, either sampled or fixed
    conf_lensing = conf["analysis"]["modelling"]["lensing"]
    ia = ["Aia", "n_Aia"] + (["bta"] if conf_lensing["extended_nla"] else [])
    fixed = parameters.get_fixed_parameters(conf)
    assert set(fixed) <= set(ia), f"only the IA parameters {ia} can be fixed on the grid, got {list(fixed)}"
    for param in ia:
        assert (param in labels) != (param in fixed), f"{param} must be either a label or in analysis.grid.fixed"
    for param, value in fixed.items():
        assert param not in grid["priors"] and param not in gaussian_priors, f"the fixed {param} has a prior"
        LOGGER.info(f"{param} is fixed at {value} on the grid")
        fiducial = conf["analysis"]["fiducial"].get(param)
        if fiducial is not None and fiducial != value:
            LOGGER.warning(f"The fiducial {param} = {fiducial} is not its fixed grid value {value}")

    # shape-noise model: get_shape_noise validates the per-field values (method/bias/fixed_bsc);
    # additionally check that a prior bias has the sampled 'sc' parameter to feed the Latin hypercube
    sn_method, sn_bias, sn_fixed_bsc, sn_survey_sys = files.get_shape_noise(conf)
    LOGGER.info(
        f"Shape-noise model: method={sn_method}, bias={sn_bias}, fixed_bsc={sn_fixed_bsc}, "
        f"survey_systematics={sn_survey_sys}"
    )
    if sn_bias == "prior":
        assert conf["analysis"]["params"].get("sc"), (
            "shape_noise bias 'prior' requires analysis.params.sc (e.g. [bsc]) to sample b_sc from the "
            "Latin hypercube"
        )
    if sn_method == "count":
        if sn_bias == "fixed":
            _check_metacal_bias_matches_forward_model(conf, sn_survey_sys)
        else:
            # the bias comes from the Latin hypercube and files.metacal_bias is never read, so there is no table
            # whose provenance could be checked -- the prior interval itself has to match the forward model
            LOGGER.warning(
                "count shape noise with bias 'prior' does not read files.metacal_bias, so the systematics "
                "consistency of the source clustering bias cannot be checked here. Make sure that "
                f"analysis.grid.priors.bsc was chosen for survey_systematics={sn_survey_sys}"
            )

    # clustering
    bg_params = conf["analysis"]["params"]["bg"]["linear"]
    conf_clustering = conf["analysis"]["modelling"]["clustering"]

    assert not (
        conf_clustering["power_law_biasing"] and conf_clustering["per_bin_biasing"]
    ), "Cannot have both power law and per bin biasing"

    assert conf_clustering["power_law_biasing"] == {"bg", "n_bg"}.issubset(
        bg_params
    ), "Power law biasing is not consistent with bg parameters"

    # one bias per lens bin, in z_bins order and named by the bin number (maglim3 -> bg3)
    per_bin_params = [f"bg{files.get_z_bin_number(z_bin)}" for z_bin in conf["survey"]["maglim"]["z_bins"]]
    assert conf_clustering["per_bin_biasing"] == (list(bg_params) == per_bin_params), (
        f"Per bin biasing is not consistent with bg parameters: expected {per_bin_params} for "
        f"survey.maglim.z_bins, got {bg_params}"
    )
    if conf_clustering["per_bin_biasing"] and conf_clustering["quadratic_biasing"]:
        assert len(conf["analysis"]["params"]["bg"]["quadratic"]) == len(
            bg_params
        ), "Quadratic per bin biasing needs one analysis.params.bg.quadratic entry per lens bin"

    if conf_clustering["stochasticity"]:
        assert conf["analysis"]["params"]["bg"]["stochasticity"] == ["rg"]
