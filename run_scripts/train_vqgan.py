import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_vqgan_256.yaml',
    'resume_from': 'work_dirs/vqgan_ffhq_256/run_6/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)