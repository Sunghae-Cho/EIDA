import numpy as np
import torch


def PCA(list_x, begin, end, plane_dim, device):
    d_model = list_x[0].shape[0]
    dtype = list_x[0].dtype

    with torch.no_grad():
        X = torch.stack(list_x[begin:end]).to(device) # list로 입력받은 벡터들을 병렬연산을 위해서 하나의 텐서로 만들기

        # 1. 주성분분석을 시작하기 전에 우선, 추정할 부분공간의 첫번째 차원은 입력된 벡터들의 평균벡터의 방향으로 설정. 
        avg = X.sum(dim=0) 
        avg = avg / torch.norm(avg)
        
        proj = X @ avg
        X = X - proj.unsqueeze(1) * avg # 벡터들의 집합에서 평균벡터 방향의 성분을 모두 제거함
        
        tensor_vectors = avg.unsqueeze(0) # vectors_plane은 추정 부분공간의 기저벡터들로 구성될 예정. 지금은 벡터 1개로 시작함.

        # 2. 입력에서 평균벡터의 성분을 제거한 상태에서, 이 분포를 잘 설명할 수 있는 부분공간의 방향들을 지금부터 주성분분석으로 추정함.
        # vectors_plane이 plane_dim 개의 벡터를 가질 때까지 입력벡터들의 분산이 큰 방향부터 방향향벡터를 1개씩 얻어내는 것을 반복하려고 함.
        # 아래의 코드는 랜덤벡터에서 시작한 r이 주방향 중 하나에 수렴하게 하는 과정이고, 수렴점이 각 루프에서 찾고자 하는 가장 큰 분산의 주성분일 확률이 100%는 아니지만 매우 높기 때문에 원하는 것이 얻어진다고 가정하고 진행함.
        for k in range(1, plane_dim):
            # 2-1. 지금까지 얻은 k차원 부분공간과 나란한 성분이 하나도 없는, 크기 1짜리 랜덤벡터 r 생성
            r = torch.from_numpy(np.random.randn(d_model).astype(np.float32)).to(dtype=dtype, device=device)
            proj = tensor_vectors @ r
            proj = proj.unsqueeze(1) * tensor_vectors
            proj = proj.sum(dim=0)
            r = r - proj
            r = r / torch.norm(r)

            # 2-2. r을 각 벡터 x에 대한 dot(x,r)*x들의 합으로 바꿔서 주방향으로 수렴하게 하는 과정
            cnt = 0
            while True:
                proj = X @ r
                s = X.T @ proj # s는 각 벡터 x에 대해 dot(x,r)*x 의 합
                var = torch.dot(r, s) # X^T * X 의 고유값(으로 수렴 중인 값)
                diff = torch.norm(var * r - s)

                r = s / torch.norm(s)
                cnt = cnt + 1
                if cnt >= 32 and diff < 0.001 * var:
                    tensor_vectors = torch.cat([tensor_vectors, r.unsqueeze(0)], dim=0) # 새로 얻어진 주방향을 평면에 추가하기
                    break

            # 2-3. 입력벡터들에서 r방향 성분을 제거하기
            proj = X @ r
            X = X - proj.unsqueeze(1) * r

    return tensor_vectors