import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_vqvae_64.yaml',
    'resume_from': None, # 'work_dirs/vqvae_ffhq_64/run_1/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)