# EIDA

실행 가능한 노트북

 - [RoBERTa-SST2-graph.ipynb](https://github.com/Sunghae-Cho/EIDA/blob/main/RoBERTa-SST2-graph.ipynb) : GLUE SST-2의 train set 데이터를 RoBERTa-base 모델에 통과시키면서 토큰 표본을 수집하여 차원 압축을 수행하고, 토큰의 원래 분포가 압축된 차원에 정사영되었을 때 성분을 얼마나 유지하는지 측정하는 코드입니다. 이 노트북을 실행하면 측정결과를 나타내는 그래프가 그려집니다.

 - [RoBERTa-SST2-train.ipynb](https://github.com/Sunghae-Cho/EIDA/blob/main/RoBERTa-SST2-train.ipynb) : 어댑터 EIDA를 적용하여 RoBERTa-base 모델을 GLUE SST-2 데이터셋으로 학습하고 accuracy를 측정하는 코드입니다. 이 노트북을 실행하면 학습된 모델이 checkpoints로 저장됩니다.

 - [GPT2-E2ENLG-graph.ipynb](https://github.com/Sunghae-Cho/EIDA/blob/main/GPT2-E2ENLG-graph.ipynb) : E2E NLG Challenge의 train set 데이터를 GPT2 모델에 통과시키면서 토큰 표본을 수집하여 차원 압축을 수행하고, 토큰의 원래 분포가 압축된 차원에 정사영되었을 때 성분을 얼마나 유지하는지 측정하는 코드입니다. 이 노트북을 실행하면 측정결과를 나타내는 그래프가 그려집니다.

 - [GPT2-E2ENLG-train.ipynb](https://github.com/Sunghae-Cho/EIDA/blob/main/GPT2-E2ENLG-train.ipynb) : 어댑터 EIDA를 적용하여 GPT2 모델을 E2E NLG Challenge 데이터셋으로 학습하고 https://github.com/tuetschek/e2e-dataset 에서 제공하는 벤치마크의 형식에 맞게 출력을 생성시키는 코드입니다. 이 노트북을 실행하면 학습된 모델이 checkpoints로 저장됩니다.


EIDA 패키지 구조

 - [EIDA/EIDA.py](https://github.com/Sunghae-Cho/EIDA/blob/main/EIDA/EIDA.py) : RoBERTa-base 파라미터들에 어댑터를 적용하기 위한 Linear_with_adapter 클래스와 GPT2 파라미터들에 어댑터를 적용하기 위한 Conv1D_with_adapter 클래스의 정의를 가지고 있습니다.

 - [EIDA/PCA.py](https://github.com/Sunghae-Cho/EIDA/blob/main/EIDA/PCA.py) : 토큰들의 표본 list을 입력받아 주성분 분석을 수행하는 함수 PCA를 가지고 있습니다. 이 함수는 얻어진 i번째 주성분벡터를 i번째 행으로 가지는 torch.tensor를 반환합니다.

 - [EIDA/reconstruct_roberta.py](https://github.com/Sunghae-Cho/EIDA/blob/main/EIDA/reconstruct_roberta.py) : 데이터가 RoBERTa-base 모델을 통과하는 과정에서 토큰 표본 추출을 수행하는 함수를 가지고 있습니다. 표본을 배열로 반환하는 forward_roberta 함수와, 표본을 지정된 경로에 파일로 저장하는 forward_roberta_with_save 함수가 있습니다.

 - [EIDA/reconstruct_gpt2.py](https://github.com/Sunghae-Cho/EIDA/blob/main/EIDA/reconstruct_gpt2.py) : 데이터가 GPT2 모델을 통과하는 과정에서 토큰 표본 추출을 수행하는 함수를 가지고 있습니다. 표본을 배열로 반환하는 forward_gpt2 함수와, 표본을 지정된 경로에 파일로 저장하는 forward_gpt2_with_save 함수가 있습니다.


그 외

 - E2E 폴더의 txt 파일들 : E2E NLG Challenge 데이터셋을 담고 있습니다.





