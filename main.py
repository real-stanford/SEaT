from environment.teleRobotEnv import TeleRobotEnv
import hydra
from omegaconf import DictConfig
from learning.srg import SRG
import numpy as np
from utils import seed_all
from time import sleep
from pathlib import Path


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_all(cfg.perception)
    # get the basic environment running
    cropped_vol_shape = np.array(cfg.env.cropped_vol_shape)
    hw = np.ceil(cropped_vol_shape / 2).astype(np.int) # half width
    srg = SRG(cfg.perception, hw)
    env = TeleRobotEnv(cfg, srg=srg, gui=True)
    env.reset_scene(Path("/tmp"))

    while True:
        env.upload_scene_to_client()
        print("Waiting for client to mainipulate the scene ...")
        while not env.update_scene_from_client(): 
            sleep(1)


if __name__ == "__main__":
    main()