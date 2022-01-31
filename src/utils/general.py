import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

import shutil

from joblib import Parallel, delayed

from fvcore.common.checkpoint import Checkpointer
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

# crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
# disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
# risk = {'1':'초기','2':'중기','3':'말기'}

# label_description = {}
# for key, value in disease.items():
#     label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
#     for disease_code in value:
#         for risk_code in risk:
#             label = f'{key}_{disease_code}_{risk_code}'
#             label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
# list(label_description.items())[:10]

# label_encoder = {key:idx for idx, key in enumerate(label_description)}
# label_decoder = {val:key for key, val in l`abel_encoder.items()}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# def predic2label(predic): 
#   if torch.is_tensor(predic): 
#     predic = predic.to_numpy()
#   else: 
#     predic = np.array(predic)

#   lst = Parallel(n_jobs=32,prefer="threads")(delayed(label_decoder)(i) for i in tqdm(predic))
#   return np.array(lst)

# def savecsv(prediclist,idx,save_path):
  
#   = np.stack(idx,prediclist)
#   pd.DataFrame()
#   return 

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import ObserverBase, FakeQuantizeBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import ObserverBase, FakeQuantizeBase

__all__ = ["Checkpointer", "PeriodicCheckpointer"]


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])

class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass

class PlantCheckpointer(Checkpointer):
    def load(
        self, path: str, checkpointables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load from the given checkpoint.
        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.
        Args:
            checkpoint (Any): checkpoint contains the weights.
        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)
            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                # Allow mismatch for uninitialized parameters
                if TORCH_VERSION >= (1, 8) and isinstance(
                    model_param, nn.parameter.UninitializedParameter
                ):
                    continue
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:

                    has_observer_base_classes = (
                        TORCH_VERSION >= (1, 8)
                        and hasattr(quantization, "ObserverBase")
                        and hasattr(quantization, "FakeQuantizeBase")
                    )
                    if has_observer_base_classes:
                        # Handle the special case of quantization per channel observers,
                        # where buffer shape mismatches are expected.
                        def _get_module_for_key(
                            model: torch.nn.Module, key: str
                        ) -> torch.nn.Module:
                            # foo.bar.param_or_buffer_name -> [foo, bar]
                            key_parts = key.split(".")[:-1]
                            cur_module = model
                            for key_part in key_parts:
                                cur_module = getattr(cur_module, key_part)
                            return cur_module

                        cls_to_skip = (
                            ObserverBase,
                            FakeQuantizeBase,
                        )
                        target_module = _get_module_for_key(self.model, k)
                        if isinstance(target_module, cls_to_skip):
                            # Do not remove modules with expected shape mismatches
                            # them from the state_dict loading. They have special logic
                            # in _load_from_state_dict to handle the mismatches.
                            continue

                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)