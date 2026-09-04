import os
import glob
import importlib
import hydra
import logging
import torch
import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


#### Configuration setup
@hydra.main(
    config_path="config",
    config_name="postprocess.yaml",
)
def set_conf(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    cfg_run = OmegaConf.load(
        hydra.utils.to_absolute_path(f"runs/{cfg.RUN_DIR}/.hydra/config.yaml")
    )
    logger.info("Original configuration:")
    logger.info(OmegaConf.to_yaml(cfg_run))
    model = importlib.import_module(cfg_run.MODEL_NAME + ".Model")

    # RNG
    torch.manual_seed(0)

    # get checkpoints
    checkpoints = list(
        sorted(
            glob.glob(f"{hydra.utils.get_original_cwd()}/runs/{cfg.RUN_DIR}/*.pth"),
            key=os.path.getmtime,
        )
    )

    # get checkpoints
    checkpoints = list(
        sorted(
            glob.glob(f"{hydra.utils.get_original_cwd()}/runs/{cfg.RUN_DIR}/*.pth"),
            key=lambda member: int((member.split("Iter_")[-1]).split(".")[0]),
        )
    )

    CHECKPOINT_FILE_FIRST = checkpoints[0]
    if cfg.CHECKPOINT_FILE == "LATEST":
        CHECKPOINT_FILE_TARGET = checkpoints[-1]
        try:
            CHECKPOINT_FILE_PREV = checkpoints[-2]
        except:
            CHECKPOINT_FILE_PREV = checkpoints[-1]
            

    else:
        CHECKPOINT_FILE_TARGET = list(
            filter(lambda x: x.endswith(cfg.CHECKPOINT_FILE), checkpoints)
        )[0]
        checkpoint_target_indx = int((CHECKPOINT_FILE_TARGET.split("Iter_")[-1]).split(".")[0])
        CHECKPOINT_FILE_PREV = f"{hydra.utils.get_original_cwd()}/runs/{cfg.RUN_DIR}/Iter_{checkpoint_target_indx-1}.pth"


    checkpoint_target_indx = int((CHECKPOINT_FILE_TARGET.split("Iter_")[-1]).split(".")[0])
    checkpoint_first_indx = int((CHECKPOINT_FILE_FIRST.split("Iter_")[-1]).split(".")[0])

    logger.info(f"Loading checkpoint file: {CHECKPOINT_FILE_TARGET}")


    m = model.SpecifiedModel.load(
        path=CHECKPOINT_FILE_TARGET,
        cfg_override={"distributed": False, "init_with_zeros": False, "MODEL_NAME": cfg_run.MODEL_NAME},
    )
    try:
        m_prev = model.SpecifiedModel.load(
            path=CHECKPOINT_FILE_PREV,
            cfg_override={"distributed": False, "init_with_zeros": False, "MODEL_NAME": cfg_run.MODEL_NAME},
        )
    except:
        logger.info(f"Previous checkpoint file not found: {CHECKPOINT_FILE_PREV}")
        m_prev = m

    logging.getLogger("DPGPModel").setLevel(30)

    pp = importlib.import_module(cfg_run.MODEL_NAME + ".PostProcess")

    pp.process(m, cfg, checkpoint_first_indx, checkpoint_target_indx)

    shock_lst = np.loadtxt((f"../../../../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/2D_shock_lst.txt"))
    
    sim_path = pp.simulate(m,m_prev, cfg, model, shock_lst)

    # iterate over checkpoints
    n_checkpoints = len(checkpoints)
    n_error_steps = 10000
    if n_checkpoints > 1:
        start_indx = max(checkpoint_first_indx,checkpoint_target_indx - n_error_steps)
        init_run = True
        for i in range(start_indx - checkpoint_first_indx,checkpoint_target_indx - checkpoint_first_indx):    
            m1 = model.SpecifiedModel.load(
                path=checkpoints[i],
                # no override, use saved params
                cfg_override={"distributed": False, "init_with_zeros": False, "MODEL_NAME": cfg_run.MODEL_NAME},
            )
            m2 = model.SpecifiedModel.load(
                path=checkpoints[i + 1],
                # no override, use saved params
                cfg_override={"distributed": False, "init_with_zeros": False, "MODEL_NAME": cfg_run.MODEL_NAME},
            )
            pp.compare(m1, m2, checkpoint_first_indx, checkpoint_target_indx, init_run, sim_path)
            init_run = False



set_conf()
