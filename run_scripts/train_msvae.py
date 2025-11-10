import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/autoencoders/train_msvae_128_high_kl.yaml',
    'resume_from': 'work_dirs/msvae_ffhq_128_high_kl/run_17/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)