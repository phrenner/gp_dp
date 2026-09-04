import torch
import hydra
from omegaconf import OmegaConf
import os
import logging
import importlib
import pathlib
from hydra.core.hydra_config import HydraConfig
import numpy as np
from mpi_dist import dist

logging.getLogger("cyipopt").setLevel(30)
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))
if rank != -1:
    logger = logging.getLogger(f"{__name__}.Rank{rank}")
else:
    logger = logging.getLogger(f"{__name__}")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#### Configuration setup
@hydra.main(
    config_path="config",
    config_name="config.yaml",
)
def set_conf(cfg):
    seed_offset = 0

    # any override parameters
    cfg = OmegaConf.to_container(cfg)
    cfg.update(
        {
            "cwd": os.getcwd(),
        }
    )

    # distributed setup
    if os.getenv("OMPI_COMM_WORLD_SIZE"):
        cfg["distributed"] = True
        dist.init_process_group(backend="mpi")
    else:
        cfg["distributed"] = False

    # RNG
    cfg["mc_seed"] = cfg["seed"] + int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    torch.manual_seed(cfg["mc_seed"])
    np.random.seed(cfg["mc_seed"] + seed_offset)

    model = importlib.import_module(cfg["MODEL_NAME"] + ".Model")

    try:
        # dynamic parameters
        cfg = importlib.import_module(
            cfg["MODEL_NAME"] + ".Params"
        ).dynamic_params(cfg)

    except FileNotFoundError:
        logger.warning(
            "No Params.py found - continuing with using only parameters from yaml file."
        )
    except AttributeError:
        logger.warning(
            "No method named `dynamic_params` found in Params.py - continuing using only parameters from yaml file."
        )

    logger.info("Running with parameters: " + str(cfg))


    save_file = None
    if cfg["STARTING_POINT"] == "NEW":
        pass
    elif cfg["STARTING_POINT"] == "LATEST":
        run_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
        model_dir = run_dir
        logger.debug(f"Searching for latest save file in: {model_dir.resolve()}")

        def _iter_num(p):
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return -1

        all_pth = list(model_dir.rglob("*.pth"))
        if all_pth:
            save_file = max(all_pth, key=_iter_num)
    else:
        save_file = cfg["STARTING_POINT"]

    if save_file:
        cfg.update({"init_with_zeros": False})
        # load
        logger.info(f"Resuming from state {save_file}")
        m = model.SpecifiedModel.load(
            path=save_file,
            cfg_override=cfg,
        )
    else:
        logger.info(f"Starting from scratch")
        m = model.SpecifiedModel(
            cfg=cfg,
        )

    for i in range(cfg.get("num_iterations", 100000)):
        #logger.info(f"Starting iteration: {i}")
        m.iterate()



set_conf()
