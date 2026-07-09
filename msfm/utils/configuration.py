from msfm.utils import logger, files

LOGGER = logger.get_logger(__file__)


def print_and_check_modeling_in_config(conf):
    LOGGER.info("Modeling choices:")
    for key, value in dict(conf["analysis"]["modelling"]).items():
        if isinstance(value, dict):
            LOGGER.info(f"{key}:")
            for k, v in value.items():
                LOGGER.info(f"  {k} = {v}")
        else:
            LOGGER.info(f"{key} = {value}")

    if conf["analysis"]["modelling"]["degrade_to_grf"]:
        LOGGER.warning(f"Degrading to Gaussian Random Field")

    # lensing
    conf_lensing = conf["analysis"]["modelling"]["lensing"]
    if conf_lensing["extended_nla"]:
        assert conf["analysis"]["params"]["ia"]["tatt"] == ["bta"]

    # shape-noise model: get_shape_noise validates the per-field values (method/bias/fixed_bsc);
    # additionally check that a prior bias has the sampled 'sc' parameter to feed the Latin hypercube
    sn_method, sn_bias, sn_fixed_bsc = files.get_shape_noise(conf)
    LOGGER.info(f"Shape-noise model: method={sn_method}, bias={sn_bias}, fixed_bsc={sn_fixed_bsc}")
    if sn_bias == "prior":
        assert conf["analysis"]["params"].get("sc"), (
            "shape_noise bias 'prior' requires analysis.params.sc (e.g. [bsc]) to sample b_sc from the "
            "Latin hypercube"
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

    assert conf_clustering["per_bin_biasing"] == {
        f"bg{i+1}" for i, _ in enumerate(conf["survey"]["maglim"]["z_bins"])
    }.issubset(bg_params), "Per bin biasing is not consistent with bg parameters"

    if conf_clustering["stochasticity"]:
        assert conf["analysis"]["params"]["bg"]["stochasticity"] == ["rg"]
