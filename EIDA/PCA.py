import os
import numpy as np
import torch

def PCA(list_x, plane_dim):
    d_model = list_x[0].shape[0]
    device = list_x[0].device
    dtype = list_x[0].dtype

    with torch.no_grad():
        X = torch.stack(list_x)

        avg = X.sum(dim=0)
        avg = avg / torch.norm(avg)
        
        proj = X @ avg
        X = X - proj.unsqueeze(1) * avg
        
        vectors_plane = avg.unsqueeze(0)
        for _ in range(1, plane_dim):
            r = torch.from_numpy(np.random.randn(d_model).astype(np.float32)).to(dtype=dtype, device=device)
            proj = vectors_plane @ r
            proj = proj.unsqueeze(1) * vectors_plane
            proj = proj.sum(dim=0)
            r = r - proj
            r = r / torch.norm(r)
            cnt = 0
            diff_prev = 0
            while True:
                proj = X @ r
                s = X.T @ proj
                val = torch.dot(r, s)
                diff_now = torch.norm(val * r - s)
                s = s / torch.norm(s)

                r = s
                cnt = cnt + 1
                if (diff_prev <= diff_now and cnt >= 8) or cnt == 64:
                    vectors_plane = torch.cat([vectors_plane, r.unsqueeze(0)], dim=0)
                    break
                diff_prev = diff_now
            proj = X @ r
            X = X - proj.unsqueeze(1) * r
            for i in range(len(list_x)):
                list_x[i] = list_x[i] - torch.dot(list_x[i], r) * r
    return vectors_plane
