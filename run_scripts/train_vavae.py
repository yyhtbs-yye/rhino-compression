import types
from rhtrain.rhino_train import main

args_dict = {
    'config': 'configs/train_vavae_256.yaml',
    'resume_from': 'work_dirs/vavae_ffhq_256/run_5/boat_state_step=35000_epoch=8.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)