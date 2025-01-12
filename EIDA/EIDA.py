import torch
from transformers.modeling_utils import Conv1D


class Linear_with_adapter(torch.nn.Module): # RoBERTa-base 모델의 torch.nn.Linear 파라미터들을 교체하기 위한 어댑터 클래스
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
            self.original_param.bias.requires_grad = True
            self.original_param.bias = self.original_param.bias.to(self.device)
        return self.original_param



class Conv1D_with_adapter(torch.nn.Module): # GPT2 모델의 Conv1D 파라미터들을 교체하기 위한 어댑터 클래스
    def __init__(
        self,
        original_param,
        A,
        C,
    ):
        super().__init__()

        param_nx = original_param.nx # A의 input 차원은 원본 가중치의 input 차원과 같게.
        param_nf = original_param.nf # C의 output 차원은 원본 가중치의 output 차원과 같게.
        adapter_nf = C.shape[0] # C는 [추정된 부분공간 차원, 원본 가중치의 input 차원] shape의 텐서로 전달됨. B의 output 차원은 이 추정된 부분공간 차원과 같게.
        adapter_nx = A.shape[0] # A는 [추정된 부분공간 차원, 원본 가중치의 output 차원] shape의 텐서로 전달됨. B의 input 차원은 이 추정된 부분공간 차원과 같게.
        assert param_nx == A.shape[1]
        assert param_nf == C.shape[1]

        self.device = original_param.weight.device
        self.dtype = original_param.weight.dtype

        with torch.no_grad():
            self.original_param = original_param
            self.original_param.weight.requires_grad = False
            self.original_param.bias.requires_grad = False

            self.A = Conv1D(nx=param_nx, nf=adapter_nx)
            self.A.weight.copy_(A.T)
            self.A.weight.requires_grad = False
            torch.nn.init.constant_(self.A.bias, 0)
            self.A.bias.requires_grad = False
            self.A.to(self.device)

            self.B = Conv1D(nx=adapter_nx, nf=adapter_nf)
            torch.nn.init.constant_(self.B.weight, 0) # B를 0으로 초기화해서, 어댑터 전체의 작용이 0으로 시작하게 됨
            torch.nn.init.constant_(self.B.bias, 0)
            self.B.bias.requires_grad = False
            self.B.to(self.device)
            
            self.C = Conv1D(nx=adapter_nf, nf=param_nf)
            self.C.weight.copy_(C)
            self.C.weight.requires_grad = False
            torch.nn.init.constant_(self.C.bias, 0)
            self.C.bias.requires_grad = False
            self.C.to(self.device)
    
    def forward(self, x):
        y = self.original_param(x)
        z = self.A(x)
        z = self.B(z)
        z = self.C(z)
        return y + z
    
    def merge(self):
        with torch.no_grad():
            self.original_param.weight = torch.nn.Parameter(self.original_param.weight + self.A.weight @ self.B.weight @ self.C.weight)
            self.original_param.weight.requires_grad = True
            self.original_param.bias.requires_grad = True
            self.original_param.to(self.device)
        return self.original_param



class Conv1D_with_adapter_for_c_attn(torch.nn.Module): # GPT2 모델의 파라미터들 중 W_Q, W_K, W_V를 합쳐놓은 c_attn에 어댑터를 달기 위한 클래스. W_Q, W_K, W_V 부분을 구분한 다음 각각에 파라미터를 달아주는 방식임.
    def __init__(
        self,
        original_param,
        A,
        C_Q,
        C_K,
        C_V
    ):
        super().__init__()
        # c_attn은 W_Q, W_K, W_V를 모아둔 가중치로, out_feature가 d_model의 3배.
        param_nx = original_param.nx
        param_nf = original_param.nf // 3
        adapter_nx = A.shape[0]
        adapter_Q_nf = C_Q.shape[0]
        adapter_K_nf = C_K.shape[0]
        adapter_V_nf = C_V.shape[0]

        self.device = original_param.weight.device
        self.dtype = original_param.weight.dtype

        with torch.no_grad():
            self.original_param = original_param
            self.original_param.weight.requires_grad = False
            self.original_param.bias.requires_grad = False

            self.A = Conv1D(nx=param_nx, nf=adapter_nx)
            self.A.weight.copy_(A.T)
            self.A.weight.requires_grad = False
            torch.nn.init.constant_(self.A.bias, 0)
            self.A.bias.requires_grad = False
            self.A.to(self.device)

            self.B_Q = Conv1D(nx=adapter_nx, nf=adapter_Q_nf)
            torch.nn.init.constant_(self.B_Q.weight, 0)
            self.C_Q = Conv1D(nx=adapter_Q_nf, nf=param_nf)
            self.C_Q.weight.copy_(C_Q)
            self.C_Q.weight.requires_grad = False
            torch.nn.init.constant_(self.C_Q.bias, 0)
            self.C_Q.bias.requires_grad = False
            self.C_Q.to(self.device)

            self.B_K = Conv1D(nx=adapter_nx, nf=adapter_K_nf)
            torch.nn.init.constant_(self.B_K.weight, 0)
            self.C_K = Conv1D(nx=adapter_K_nf, nf=param_nf)
            self.C_K.weight.copy_(C_K)
            self.C_K.weight.requires_grad = False
            torch.nn.init.constant_(self.C_K.bias, 0)
            self.C_K.bias.requires_grad = False
            self.C_K.to(self.device)
    
            self.B_V = Conv1D(nx=adapter_nx, nf=adapter_V_nf)
            torch.nn.init.constant_(self.B_V.weight, 0)
            self.C_V = Conv1D(nx=adapter_V_nf, nf=param_nf)
            self.C_V.weight.copy_(C_V)
            self.C_V.weight.requires_grad = False
            torch.nn.init.constant_(self.C_V.bias, 0)
            self.C_V.bias.requires_grad = False
            self.C_V.to(self.device)

    def forward(self, x):
        y = self.original_param(x)
        z_Q = self.C_Q(self.B_Q(self.A(x)))
        z_K = self.C_K(self.B_K(self.A(x)))
        z_V = self.C_V(self.B_V(self.A(x)))
        return y + torch.cat((z_Q, z_K, z_V), dim=2)
    
    def merge(self):
        with torch.no_grad():
            self.original_param.weight = torch.nn.Parameter(self.original_param.weight + torch.cat(((self.A.weight @ self.B_Q.weight @ self.C_Q.weight), (self.A.weight @ self.B_K.weight @ self.C_K.weight), (self.A.weight @ self.B_V.weight @ self.C_V.weight)), dim=1))
            self.original_param.weight.requires_grad = True
            self.original_param.bias.requires_grad = True
            self.original_param.to(self.device)
        return self.original_param