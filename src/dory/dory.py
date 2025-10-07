from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def main(path: str):
    """
    torchrun --nproc-per-node 8 src/dory/dory.py --config configs/a3b.yaml
    """
    cfg = parse_args_and_load_config(path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()