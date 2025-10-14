
import torch
import torch_xla.core.xla_model as xm

"""
manual euler sampling with v prediction -- torchdiffeq doesn't support automatic xm.mark_step() and has problem in XLA devices
"""
def manual_sample(stage2_model, z_init, y, schedule, device: torch.device, model_kwargs = None) -> torch.Tensor:
    zt = z_init
    with torch.no_grad():
        iterator = range(len(schedule) - 1, 0, -1)
        for i in iterator:
            t_cur = torch.full((zt.shape[0],), schedule[i], device=device, dtype=zt.dtype)
            t_prev = torch.full((zt.shape[0],), schedule[i - 1], device=device, dtype=zt.dtype)
            v_pred = stage2_model(zt, t_cur, y, **(model_kwargs or {}))
            delta = (t_prev - t_cur).view(-1, 1, 1, 1)
            zt = zt + delta * v_pred
            xm.mark_step() # trigger compilation on XLA devices
    return zt

def make_timesteps(num_steps: int, t_min: float, t_max: float, shift: float) -> torch.Tensor:
    schedule = torch.linspace(t_min, t_max, steps=num_steps, dtype=torch.float32)
    schedule = shift * schedule / (1 + (shift - 1) * schedule)
    return schedule