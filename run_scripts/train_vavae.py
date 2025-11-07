import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_vavae_256_high_kl.yaml',
    'resume_from': 'work_dirs/vavae_ffhq_256_high_kl/run_1/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)