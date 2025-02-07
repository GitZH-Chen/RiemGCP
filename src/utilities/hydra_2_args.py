import os
import torch as th
from omegaconf import DictConfig, OmegaConf



def parse_cfg(args, cfg: DictConfig):
    # Function to recursively set attributes, keeping only the final key name
    def set_attributes_from_dict(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                # If the value is a dict, continue to extract its values
                set_attributes_from_dict(target, value)
            else:
                # Directly set the attribute on the target
                setattr(target, key, value)

    # Convert Hydra config to a nested dictionary and then flatten it
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_attributes_from_dict(args, cfg_dict)

    # set lr_params
    args.lr_params=[args.lr_params]

    args.benchmark=args.name
    if args.name.startswith('CUB_200_2011'):
        args.abbrevated_dataset = 'cub'
    elif args.name.startswith('fgvc-aircraft'):
        args.abbrevated_dataset = 'air'
    elif args.name.startswith('fgvc-cars'):
        args.abbrevated_dataset = 'cars'
    elif args.name.startswith('ILSVRC2012'):
        args.abbrevated_dataset = 'ImgNet'
    else:
        raise KeyError("=> undefined  benchmark {}".format(args.benchmark))

    args.modelname,args.modeldir = get_model_name(args)
    return args

def get_model_name(args):
    if args.modeldir == None:
        lr_params_str = f'lp_{args.lr_params}'
        if args.representation == 'MPNCOV':
            modeldir = os.path.join('models',
                                         f'{args.abbrevated_dataset}-{args.arch}-{args.representation}-{args.transformed_mode}-lr{args.lr}-wd{args.weight_decay}-cf{args.classifier_factor}-{lr_params_str}-bs{args.batch_size}-sd{args.seed}')
        elif args.representation == 'LCMCOV':
            modeldir = os.path.join('models',
                                         f'{args.abbrevated_dataset}-{args.arch}-{args.representation}-eps{args.eps}-lr{args.lr}-wd{args.weight_decay}-cf{args.classifier_factor}-{lr_params_str}-bs{args.batch_size}-sd{args.seed}')

    modelname = modeldir.split('/')[-1]
    return modelname,modeldir







