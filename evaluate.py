# This script will contain code for evaluation of various things:
# - Data:
# - Segmentation Model
# - Shape Completion Model
# Each evaluation will generate a webpage containing
# - summary of qualitative metrics
# - visualization of predictions on test set

import hydra
from omegaconf import DictConfig
from evaluate.evaluate_data import evaluate_data, evaluate_data_tn, evaluate_data_depth2orient, evaluate_prepared_data, evaluate_vol_match
from evaluate.evaluate_model import evaluate_tn_model, evaluate_seg, evaluate_sc_model, evaluate_vol_match_model
from utils import seed_all_int


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_all_int(cfg.seeds.test)
    if cfg.evaluate.name == "data":
        evaluate_data(cfg)
    elif cfg.evaluate.name == "sc_model":
        evaluate_sc_model(cfg)
    elif cfg.evaluate.name == "seg":
        evaluate_seg(cfg)
    elif cfg.evaluate.name == "tn":
        evaluate_data_tn(cfg)
    elif cfg.evaluate.name == "depth2orient":
        evaluate_data_depth2orient(cfg)
    elif cfg.evaluate.name == "tn_model":
        evaluate_tn_model(cfg)
    elif cfg.evaluate.name == "vol_match" or cfg.evaluate.name == "vol_match_6DoF":
        evaluate_vol_match(cfg)
    elif cfg.evaluate.name == "vol_match_6DoF_model":
        evaluate_vol_match_model(cfg)
    elif cfg.evaluate.name == "prepared_data":
        evaluate_prepared_data(cfg)


if __name__ == "__main__":
    main()
