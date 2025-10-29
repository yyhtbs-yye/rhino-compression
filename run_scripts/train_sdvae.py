import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_sdvae_256.yaml',
    'resume_from': 'work_dirs/sdvae_ffhq_256/run_3/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)