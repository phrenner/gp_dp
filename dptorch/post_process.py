import os
import glob
import importlib
import hydra
import logging
import logging.config
import torch
from DPGPModel import DPGPModel
from DPGPIpoptModel import DPGPIpoptModel
import cyipopt
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

    if cfg.CHECKPOINT_FILE == "LATEST":
        CHECKPOINT_FILE = checkpoints[-1]
        try:
            CHECKPOINT_FILE_PREV = checkpoints[-2]
        except:
            CHECKPOINT_FILE_PREV = checkpoints[-1]
            
        checkpoint_indx = int((CHECKPOINT_FILE.split("Iter_")[-1]).split(".")[0])

    else:
        CHECKPOINT_FILE = list(
            filter(lambda x: x.endswith(cfg.CHECKPOINT_FILE), checkpoints)
        )
        checkpoint_indx = int((CHECKPOINT_FILE[0].split("Iter_")[-1]).split(".")[0])
        CHECKPOINT_FILE_PREV = f"{hydra.utils.get_original_cwd()}/runs/{cfg.RUN_DIR}/Iter_{checkpoint_indx-1}.pth"

        if not CHECKPOINT_FILE:
            raise FileNotFoundError("Specified checkpoint file not found")
        else:
            CHECKPOINT_FILE = CHECKPOINT_FILE[0]
            CHECKPOINT_FILE_PREV = CHECKPOINT_FILE[0]

    logger.info(f"Loading checkpoint file: {CHECKPOINT_FILE}")

    checkpoint_indx = int((CHECKPOINT_FILE.split("Iter_")[-1]).split(".")[0])
    checkpoint_indx_start = int((CHECKPOINT_FILE.split("Iter_")[0]).split(".")[0])

    # load the specified model
    m = model.SpecifiedModel.load(
        path=CHECKPOINT_FILE,
        # no override, use saved params
        cfg_override={"distributed": False, "init_with_zeros": False},
    )
    try:
        m_prev = model.SpecifiedModel.load(
            path=CHECKPOINT_FILE_PREV,
            # no override, use saved params
            cfg_override={"distributed": False, "init_with_zeros": False},
        )
    except:
        m_prev = m

    logging.getLogger("DPGPModel").setLevel(30)

    # call the post_processing script for the model
    pp = importlib.import_module(cfg_run.MODEL_NAME + ".PostProcess")

    #pp.logger.setLevel(logging.WARNING)

    pp.process(m, cfg, checkpoint_indx_start, checkpoint_indx)
    
    err_out = open(f"V_func_error_{checkpoint_indx_start}_{checkpoint_indx}.txt", 'w')
    err_out.write("#T1 T2 L2 LInf \n")   
    err_out.close()    
    # iterate over checkpoints
    n_checkpoints = len(checkpoints)
    if n_checkpoints > 1:
        start_indx = max(0,n_checkpoints - 100)
        for i in range(start_indx,n_checkpoints-1):    
            m1 = model.SpecifiedModel.load(
                path=checkpoints[i],
                # no override, use saved params
                cfg_override={"distributed": False, "init_with_zeros": False},
            )
            m2 = model.SpecifiedModel.load(
                path=checkpoints[i + 1],
                # no override, use saved params
                cfg_override={"distributed": False, "init_with_zeros": False},
            )
            pp.compare(m1, m2, cfg)

    pp.simulate(m,m_prev, cfg, model)


set_conf()
