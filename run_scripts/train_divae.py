import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_divae_256.yaml',
    'resume_from': None,
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)