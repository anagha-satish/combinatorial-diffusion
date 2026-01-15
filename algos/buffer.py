import random
import torch
from collections import deque
from typing import Dict, Any, Optional

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Dict[str, Any]):
        self.buffer.append(transition)

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        replace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        n = len(self.buffer)
        if n == 0:
            raise ValueError("empty replay buffer")

        if replace or batch_size > n:
            idxs = [random.randrange(n) for _ in range(batch_size)]
            batch = [self.buffer[i] for i in idxs]
        else:
            batch = random.sample(self.buffer, batch_size)

        out: Dict[str, torch.Tensor] = {}
        for key in batch[0]:
            t = torch.stack([torch.as_tensor(x[key]) for x in batch], dim=0)

            if device is not None:
                t = t.to(device)

            if key in ("s", "s_next"):
                t = t.float()
            elif key in ("a", "mask", "mask_next"):
                t = (t.float() > 0.5).float()
            elif key == "done":
                t = (t.float() > 0.5).float()
            elif key == "r":
                t = t.float()

            out[key] = t

        return out

    def __len__(self):
        return len(self.buffer)







