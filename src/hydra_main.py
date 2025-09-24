import hydra
from omegaconf import DictConfig
from .train import main_train
from .eval import main_eval  

@hydra.main(config_path="../configs", config_name="defaults", version_base="1.3")
def main(cfg: DictConfig):
    mode = cfg.get("mode","train")
    if mode == "train":
        main_train(cfg)
    elif mode == "eval":
        main_eval(cfg)
    else:
        raise ValueError(f"Unknown mode {mode}")

if __name__ == "__main__":
    main()
