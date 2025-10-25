import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_var_vqvae.yaml',
    'resume_from': 'work_dirs/varvqvae_ffhq_256/run_36/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)