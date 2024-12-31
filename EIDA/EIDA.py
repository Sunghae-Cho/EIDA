import torch

class Linear_with_adapter(torch.nn.Module):
    def __init__(
        self,
        original_param,
        A,
        C,
    ):
        super().__init__()

        param_in_features = original_param.in_features # A의 input 차원은 원본 가중치의 input 차원과 같게.
        param_out_features = original_param.out_features # C의 output 차원은 원본 가중치의 output 차원과 같게.
        adapter_out_features = C.shape[0] # C는 [추정된 부분공간 차원, 원본 가중치의 input 차원] shape의 텐서로 전달됨. B의 output 차원은 이 추정된 부분공간 차원과 같게.
        adapter_in_features = A.shape[0] # A는 [추정된 부분공간 차원, 원본 가중치의 output 차원] shape의 텐서로 전달됨. B의 input 차원은 이 추정된 부분공간 차원과 같게.
        assert param_in_features == A.shape[1]
        assert param_out_features == C.shape[1]

        self.device = original_param.weight.device
        self.dtype = original_param.weight.dtype

        with torch.no_grad():
            self.original_param = original_param
            self.original_param.weight.requires_grad = False
            if self.original_param.bias is not None:
                self.original_param.bias.requires_grad = False
            self.A = torch.nn.Linear(in_features=param_in_features, out_features=adapter_in_features, bias=False, device=self.device, dtype=self.dtype) 
            self.A.weight.copy_(A)
            self.A.weight.requires_grad = False
            self.B = torch.nn.Linear(in_features=adapter_in_features, out_features=adapter_out_features, bias=False, device=self.device, dtype=self.dtype)
            torch.nn.init.constant_(self.B.weight, 0) # B를 0으로 초기화해서, 어댑터 전체의 작용이 0으로 시작하게 됨
            self.C = torch.nn.Linear(in_features=adapter_out_features, out_features=param_out_features, bias=False, device=self.device, dtype=self.dtype)
            self.C.weight.copy_(C.T)
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


class Linear_with_adapter_for_gpt2_c_attn(torch.nn.Module):
    def __init__(
        self,
        original_param,
        A_Q,
        C_Q,
        A_K,
        C_K,
        A_V,
        C_V
    ):
        super().__init__()
        # c_attn은 W_Q, W_K, W_V를 모아둔 가중치로, out_feature가 d_model의 3배.
        param_in_features = original_param.in_features
        param_out_features = original_param.out_features // 3
        adapter_Q_out_features = C_Q.shape[0]
        adapter_Q_in_features = A_Q.shape[0]
        adapter_K_out_features = C_K.shape[0]
        adapter_K_in_features = A_K.shape[0]
        adapter_V_out_features = C_V.shape[0]
        adapter_V_in_features = A_V.shape[0]

        self.device = original_param.weight.device
        self.dtype = original_param.weight.dtype

        with torch.no_grad():
            self.original_param = original_param
            self.original_param.weight.requires_grad = False
            if self.original_param.bias is not None:
                self.original_param.bias.requires_grad = False

            self.A_Q = torch.nn.Linear(in_features=param_in_features, out_features=adapter_Q_in_features, bias=False, device=self.device, dtype=self.dtype)
            self.A_Q.weight.copy_(A_Q)
            self.A_Q.weight.requires_grad = False
            self.B_Q = torch.nn.Linear(in_features=adapter_Q_in_features, out_features=adapter_Q_out_features, bias=False, device=self.device, dtype=self.dtype)
            torch.nn.init.constant_(self.B_Q.weight, 0)
            self.C_Q = torch.nn.Linear(in_features=adapter_Q_out_features, out_features=param_out_features, bias=False, device=self.device, dtype=self.dtype)
            self.C_Q.weight.copy_(C_Q.T)
            self.C_Q.weight.requires_grad = False

            self.A_K = torch.nn.Linear(in_features=param_in_features, out_features=adapter_K_in_features, bias=False, device=self.device, dtype=self.dtype)
            self.A_K.weight.copy_(A_K)
            self.A_K.weight.requires_grad = False
            self.B_K = torch.nn.Linear(in_features=adapter_K_in_features, out_features=adapter_K_out_features, bias=False, device=self.device, dtype=self.dtype)
            torch.nn.init.constant_(self.B_K.weight, 0)
            self.C_K = torch.nn.Linear(in_features=adapter_K_out_features, out_features=param_out_features, bias=False, device=self.device, dtype=self.dtype)
            self.C_K.weight.copy_(C_K.T)
            self.C_K.weight.requires_grad = False
    
            self.A_V = torch.nn.Linear(in_features=param_in_features, out_features=adapter_V_in_features, bias=False, device=self.device, dtype=self.dtype)
            self.A_V.weight.copy_(A_V)
            self.A_V.weight.requires_grad = False
            self.B_V = torch.nn.Linear(in_features=adapter_V_in_features, out_features=adapter_V_out_features, bias=False, device=self.device, dtype=self.dtype)
            torch.nn.init.constant_(self.B_V.weight, 0)
            self.C_V = torch.nn.Linear(in_features=adapter_V_out_features, out_features=param_out_features, bias=False, device=self.device, dtype=self.dtype)
            self.C_V.weight.copy_(C_V.T)
            self.C_V.weight.requires_grad = False

    def forward(self, x):
        y = self.original_param(x)
        z_Q = self.C_Q(self.B_Q(self.A_Q(x)))
        z_K = self.C_K(self.B_K(self.A_K(x)))
        z_V = self.C_V(self.B_V(self.A_V(x)))
        return y + torch.cat((z_Q, z_K, z_V), dim=1)
    
    def merge(self):
        with torch.no_grad():
            self.original_param.weight = torch.nn.Parameter(self.original_param.weight + torch.cat((self.C_Q.weight @ self.B_Q.weight @ self.A_Q.weight), (self.C_K.weight @ self.B_K.weight @ self.A_K.weight), (self.C_V.weight @ self.B_V.weight @ self.A_V.weight), dim=0))
            self.original_param.weight.requires_grad = True
            self.original_param.weight = self.original_param.weight.to(self.device)
            if self.original_param.bias is not None:
                self.original_param.bias.requires_grad = True
                self.original_param.bias = self.original_param.bias.to(self.device)
        return self.original_param