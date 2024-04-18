import hydra
import model
from methods import *
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    m = model.SIR(cfg)
    m.run(runge_kutta_4th_order)
    m.plot()



if __name__ == "__main__":
    my_app()