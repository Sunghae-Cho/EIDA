import torch

class Linear_with_adapter(torch.nn.Module):
    def __init__(
        self,
        original_param,
        in_plane_proj,
        out_plane_proj,
    ):
        super().__init__()

        param_in_features = original_param.in_features
        param_out_features = original_param.out_features
        adapter_out_features = out_plane_proj.shape[0]
        adapter_in_features = in_plane_proj.shape[0]
        assert param_in_features == in_plane_proj.shape[1]
        assert param_out_features == out_plane_proj.shape[1]
        self.device = original_param.weight.device
        self.dtype = original_param.weight.dtype
        with torch.no_grad():
            self.original_param = original_param
            self.original_param.weight.requires_grad = False
            if self.original_param.bias is not None:
                self.original_param.bias.requires_grad = False
            self.A = torch.nn.Linear(param_in_features, adapter_in_features, bias=False, device=self.device, dtype=self.dtype)
            self.A.weight.copy_(in_plane_proj)
            self.A.weight.requires_grad = False
            self.B = torch.nn.Linear(adapter_in_features, adapter_out_features, bias=False, device=self.device, dtype=self.dtype)
            torch.nn.init.constant_(self.B.weight, 0)
            self.C = torch.nn.Linear(adapter_out_features, param_out_features, bias=False, device=self.device, dtype=self.dtype)
            self.C.weight.copy_(out_plane_proj.T)
            self.C.weight.requires_grad = False
    
    def forward(self, x):
        y = self.original_param(x)
        z = self.A(x)
        z = self.B(z)
        z = self.C(z)
        return y + z
    
    def merge(self):
        with torch.no_grad():
            self.original_param.weight = torch.nn.Parameter(self.original_param.weight + self.C.weight @ self.B.weight @ self.A.weight)
            self.original_param.weight.requires_grad = True
            self.original_param.weight = self.original_param.weight.to(self.device)
            if self.original_param.bias is not None:
                self.original_param.bias.requires_grad = True
                self.original_param.bias = self.original_param.bias.to(self.device)
        return self.original_param