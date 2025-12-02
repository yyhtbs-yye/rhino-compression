import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/autoencoders/train_msvae_128_high_kl.yaml',
    'resume_from': None
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)