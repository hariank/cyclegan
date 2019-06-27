import numpy as np
import torch


class ImageBuffer():
    def __init__(self, cap=50):
        self.cap = cap
        self.images = []

    def insert_and_sample(self, imgs, n_sample):
        ret = []
        for img in imgs:
            batched_im = torch.unsqueeze(img, 0)
            if np.random.uniform() > 0.5:  # randomly insert into buffer
                if len(self.images) < self.cap:
                    self.images.append(batched_im)
                else:
                    self.images[np.random.randint(len(self.images))] = batched_im
            if np.random.uniform() > 0.5 and len(self.images):  # randomly replace with sampled img
                batched_im = self.images[np.random.randint(len(self.images))]
            ret.append(batched_im)
        return torch.cat(ret)
