import math
import warnings
from typing import Optional, cast, List
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, _enable_get_lr_call


class Lin_incr_LRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr: float, max_steps: int, current_step: int = None):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.max_lr / self.max_steps * (1 + current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class Lin_incr_offset_LRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr: float, max_steps: int, start_step: int, current_step: int = None):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.start_step = start_step
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.max_lr / self.max_steps * (1 + current_step - self.start_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class PolyLRScheduler_offset(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        start_step: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        current_step = current_step - self.start_step
        if current_step <= 0:
            current_step = 0

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

class CosineAnnealingWarmRestarts_Offset(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        offset=0
    ):
        self.offset = offset
        super().__init__(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    def _get_lr(self, base_lr):
        # same logic as cosine schedule, but shift epoch by offset
        cycle_progress = (self.last_epoch - self.offset - self.T_cur) / self.T_i
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cycle_progress)) / 2

    def get_lr(self):
        if self.last_epoch < self.offset:
            # Before offset: return base LR unchanged
            return list(self.base_lrs)
        return [self._get_lr(base_lr) for base_lr in self.base_lrs]


class CosineAnnealingLR_offset(CosineAnnealingLR):
    def __init__(
        self, optimizer: Optimizer, T_max: int, eta_min=0, last_epoch=-1, verbose="deprecated", offset: int = 0
    ):
        self.offset = offset
        super().__init__(
            optimizer,
            T_max,
            eta_min,
            last_epoch,
            verbose,
        )

    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.offset) / (self.T_max - self.offset)))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):

        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch
            values = cast(List[float], self._get_closed_form_lr())

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            if isinstance(param_group["lr"], Tensor):
                lr_val = lr.item() if isinstance(lr, Tensor) else lr  # type: ignore[attr-defined]
                param_group["lr"].fill_(lr_val)
            else:
                param_group["lr"] = lr

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]


class PolyLRScheduler_offset_min(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        start_step: int,
        min_lr: float = 0.0,
        exponent: float = 0.9,   
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.exponent = exponent
        self.min_lr = min_lr          
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # Adjust step based on offset
        current_step = current_step - self.start_step
        if current_step <= 0:
            current_step = 0

        # Polynomial decay
        poly_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent

        # Clamp to min_lr
        new_lr = max(self.min_lr, poly_lr)

        # Apply LR to optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

class LinearWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, eta_min=0, hold_offset=0, last_epoch=-1):
        """
        T_0: length of each linear decay cycle (in epochs)
        eta_min: minimum learning rate at end of cycle
        hold_offset: number of epochs/steps to hold LR constant at the initial base_lr
        """
        self.T_0 = T_0
        self.eta_min = eta_min
        # --- NEW PARAMETER ---
        self.hold_offset = hold_offset
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Check for the HOLD PHASE
        if self.last_epoch < self.hold_offset:
            # Return the initial base_lr (MAX_LR) for the hold duration
            return self.base_lrs 
        
        # 2. Calculate the effective epoch for cycling
        # Cycling only starts after the hold phase is complete
        effective_epoch = self.last_epoch - self.hold_offset
        
        # 3. Calculate position within the current cycle
        cycle_epoch = effective_epoch % self.T_0
        
        # 4. Calculate the linear decay factor
        if self.T_0 <= 1:
            decay_factor = 0.0
        else:
            # Use T_0 - 1 in the denominator to ensure MIN_LR is exactly hit
            decay_factor = (1 - cycle_epoch / (self.T_0 - 1)) 
    
        # 5. Return the calculated LR
        return [
            self.eta_min + (base_lr - self.eta_min) * decay_factor
            for base_lr in self.base_lrs
        ]
    

class CyclicalCosineLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, step_size, hold_offset=0, last_epoch=-1):
        """
        Implements a smooth triangular cycle using cosine interpolation.
        
        max_lr: Upper boundary of the learning rate (base_lr is set to this).
        min_lr: Lower boundary of the learning rate.
        step_size: Number of steps/epochs for ONE HALF cycle (e.g., max_lr to min_lr).
        hold_offset: number of epochs/steps to hold LR constant at the initial max_lr.
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size = step_size
        self.hold_offset = hold_offset
        
        optimizer.param_groups[0]['lr'] = max_lr 
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Check for the HOLD PHASE
        if self.last_epoch < self.hold_offset:
            # LR is fixed at the initial max_lr during the offset
            return [self.max_lr] * len(self.base_lrs)
        
        # 2. Calculate effective epoch after the hold
        effective_epoch = self.last_epoch - self.hold_offset
        
        # Total length of the full cycle is 2 * step_size
        cycle_length = 2 * self.step_size
        
        # Position in current cycle (0 to cycle_length - 1)
        cycle_position = effective_epoch % cycle_length
        
         
        if cycle_position < self.step_size:
         
            decay_fraction = cycle_position / self.step_size
            cosine_scale = 0.5 * (1 + math.cos(math.pi * decay_fraction)) 
   
        else:
          
            reheat_position = cycle_position - self.step_size
            reheat_fraction = reheat_position / self.step_size
            cosine_scale = 0.5 * (1 - math.cos(math.pi * reheat_fraction))
    
        lr_range = self.max_lr - self.min_lr
        
        return [
            self.min_lr + lr_range * cosine_scale
            for base_lr in self.base_lrs
        ]