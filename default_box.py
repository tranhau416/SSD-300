import torch

from lib import *

cfg = {
    "num_classes": 21, # VOC data include 20 class + 1 class backgroud
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], #Trọng số cho source1 ->source6
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 10, 300], # size  of default box
    "min_size": [30, 60, 111, 162, 213, 264], # size  of default box
    "max_size": [60, 111, 162, 213, 264, 315], # size  of default box
    "aspect_ratio": [[2], [2, 3], [2, 3], [2, 3], [2], [2]] #Tỉ lệ thay đổi với default box
}

class DefaultBox():
    def __init__(self, cfg):
        self.image_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratio = cfg["aspect_ratio"]
        self.steps = cfg["steps"]

    def create_dbox(self):
        dbox_lists =[]
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k] # Size of grid

                #unit center
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k

                #small box
                s_k = self.min_size[k] / self.image_size
                dbox_lists += (cx, cy, s_k, s_k)

                #big box
                s_k_ = sqrt(s_k*(self.max_size[k] / self.image_size))
                dbox_lists += (cx, cy, s_k_, s_k_)
                for ar in self.aspect_ratio[k]:
                    dbox_lists += (cx, cy, s_k * sqrt(ar), s_k / sqrt(ar))
                    dbox_lists += (cx, cy, s_k / sqrt(ar), s_k * sqrt(ar))
        output = torch.Tensor(dbox_lists).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output

if __name__ == "__main__":
    dbox = DefaultBox(cfg)
    dbox_list = dbox.create_dbox()
    # print(dbox_list)
    print(pd.DataFrame(dbox_list.numpy()))


