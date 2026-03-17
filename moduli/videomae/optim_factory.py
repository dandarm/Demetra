# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import json

import torch
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch import optim as optim

try:
    from apex.optimizers import FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def _strip_known_prefixes(var_name):
    prefixes = ("module.", "_orig_mod.", "backbone.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if var_name.startswith(p):
                var_name = var_name[len(p):]
                changed = True
    return var_name


def get_num_layer_for_vit(var_name, num_max_layer):
    var_name = _strip_known_prefixes(var_name)
    if var_name.startswith("encoder."):
        var_name = var_name[len("encoder."):]

    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("decoder.") or var_name.startswith("encoder_to_decoder"):
        return num_max_layer - 1
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model,
                         weight_decay=1e-5,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.endswith(
                ".scale") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # TODO: commento per evitare logging troppo verboso
    #print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args,
                     model,
                     get_num_layer=None,
                     get_layer_scale=None,
                     filter_bias_and_bn=True,
                     skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip,
                                          get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(
        ), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(
            parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(
            parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def create_split_adamw_optimizer(args, model):
    decoder_lr = float(args.decoder_lr)
    encoder_lr = float(args.encoder_lr)
    if decoder_lr <= 0 or encoder_lr <= 0:
        raise ValueError(
            f"decoder_lr and encoder_lr must be > 0, got decoder_lr={decoder_lr}, encoder_lr={encoder_lr}"
        )

    encoder_depth = (
        int(model.encoder.get_num_layers())
        if hasattr(model, "encoder") and hasattr(model.encoder, "get_num_layers")
        else 0
    )
    decoder_depth = (
        int(model.decoder.get_num_layers())
        if hasattr(model, "decoder") and hasattr(model.decoder, "get_num_layers")
        else 0
    )

    layer_decay = float(getattr(args, "layer_decay", 1.0) or 1.0)
    if layer_decay <= 0:
        raise ValueError(f"layer_decay must be > 0, got {layer_decay}")

    encoder_scales = [layer_decay ** (encoder_depth - i + 1) for i in range(encoder_depth + 2)]
    def _encoder_layer_id(name):
        name = _strip_known_prefixes(name)
        if name.startswith("encoder."):
            name = name[len("encoder."):]
        if name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        if name.startswith("patch_embed"):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return encoder_depth + 1

    def _decoder_layer_id(name):
        name = _strip_known_prefixes(name)
        if name.startswith("decoder."):
            stripped = name[len("decoder."):]
            if stripped.startswith("blocks."):
                return int(stripped.split(".")[1]) + 1
            return decoder_depth + 1
        if name.startswith("encoder_to_decoder") or name in ("mask_token", "pos_embed"):
            return 0
        return decoder_depth + 1

    parameter_group_vars = {}
    encoder_param_count = 0
    decoder_param_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        clean_name = _strip_known_prefixes(name)

        if clean_name.startswith("encoder."):
            branch = "encoder"
            layer_id = _encoder_layer_id(clean_name)
            base_lr = encoder_lr
            base_scale = encoder_lr / decoder_lr
            layer_scale = encoder_scales[layer_id]
            encoder_param_count += 1
            group_name = f"{branch}_layer_{layer_id}"
        else:
            branch = "decoder"
            layer_id = 0
            base_lr = decoder_lr
            base_scale = 1.0
            layer_scale = 1.0
            decoder_param_count += 1
            group_name = branch
        if group_name not in parameter_group_vars:
            parameter_group_vars[group_name] = {
                "params": [],
                "lr": base_lr * layer_scale,
                "lr_scale": base_scale * layer_scale,
                "weight_decay": float(args.weight_decay),
                "group_name": group_name,
            }
        parameter_group_vars[group_name]["params"].append(param)

    if encoder_param_count == 0:
        raise ValueError("No encoder parameters found for split AdamW optimizer.")
    if decoder_param_count == 0:
        raise ValueError("No decoder-side parameters found for split AdamW optimizer.")

    opt_args = dict(
        lr=decoder_lr,
        weight_decay=float(args.weight_decay),
    )
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print(
        "optimizer settings:",
        {
            **opt_args,
            "encoder_lr": encoder_lr,
            "decoder_lr": decoder_lr,
            "layer_decay": layer_decay,
            "encoder_group_count": sum(1 for k in parameter_group_vars if k.startswith("encoder_")),
            "decoder_group_count": sum(1 for k in parameter_group_vars if k == "decoder" or k.startswith("decoder_")),
            "encoder_params": encoder_param_count,
            "decoder_params": decoder_param_count,
        },
    )
    optimizer = optim.AdamW(list(parameter_group_vars.values()), **opt_args)
    return optimizer
