from collections import defaultdict
from typing import List, Dict
import torch



class ListCollator():
    def __call__(self, x: List[Dict]) -> Dict:
        out = defaultdict(list)
        for item in x:
            for k, v in item.items():
                out[k].append(v)

        # for k, v in out.items():
        #     if torch.is_tensor(v[0]):
        #         out[k] = torch.stack(v)

        return out
