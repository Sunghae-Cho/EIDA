import os
import torch
import random


def forward_gpt2(model, dataset_input_ids, dataset_labels, begin, end, batch_size=2, max_length=128, p=0.01):
    # args:
    # model: 허깅페이스의 "gpt2" 모델
    # dataset_input_ids, dataset_attention_mask: tokenize된 데이터셋 전체
    # begin, end: 데이터셋 안에서 표본추출에 이용할 구간의 시작 index와 끝 index
    # p: 각 token을 표본으로 뽑을 확률. 당연히 PAD토큰은 추출에서 제외함

    device = model.device

    sample_inputs = [[] for _ in range(12*4)]
    # 파라미터(c_attn, W_O, W_fc1, W_fc2)들의 input에 해당하는 token representation space의 index:
    # 4*l+0: layer[l] 에서 c_attn의 input의 위치 (768차원) (l = 0, 1, ..., 11)
    # 4*l+1: layer[l] 에서 W_O의 input의 위치 (768차원)
    # 4*l+2: layer[l] 에서 W_fc1의 input의 위치 (768차원)
    # 4*l+3: layer[l] 에서 W_fc2의 input의 위치 (3072차원)

    position_ids = torch.arange(0, max_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, max_length) # 각 시퀀스별로, 0에서 시작해서 1씩 증가하는 배열
    for k in range(begin, end, batch_size):
        input_ids = dataset_input_ids[k:k+batch_size].clone().to(device)
        labels = dataset_labels[k:k+batch_size].clone().to(device)
        
        valid_indices = labels != -100

        loss = torch.tensor(0.0, device=device)

        started = False
        for i in range(1, max_length): # auto-regressive decoding loop
            if labels[:, i].sum() == -100 * batch_size:
                if not started: # 배치 안의 모든 시퀀스가 context 영역에 있으면 통과
                    continue
                else: # 배치 안의 모든 시퀀스가 padding 부분에 들어오면 decoding loop 종료
                    break
            started = True

            causal_input_ids = input_ids[:, 0:i].clone().to(device)
            causal_position_ids = position_ids[:, 0:i].clone().to(device)
            # 이 과정에서 KV caching은 이용하지 않음

            inputs_embeds = model.transformer.wte(causal_input_ids) # Word Token Embeddings: vocab_sz=50257 가지의 토큰을 d_model=768 차원 벡터로 변환
            position_embeds = model.transformer.wpe(causal_position_ids)
            hidden_states = inputs_embeds + position_embeds
            hidden_states = model.transformer.drop(hidden_states)

            for l in range(len(model.transformer.h)): # block 12개에 대한 반복문
                block = model.transformer.h[l]

                # Self-Attention 시작
                residual = hidden_states
                hidden_states = block.ln_1(hidden_states) # layer normalization
                
                query, key, value = block.attn.c_attn(hidden_states).split(768, dim=2) 
                # GPT2는 c_attn이라는 행렬에다 W_Q, W_K, W_V를 합쳐놨음

                query = block.attn._split_heads(query, 12, 64) 
                key = block.attn._split_heads(key, 12, 64)
                value = block.attn._split_heads(value, 12, 64)
                # d_model=768은 12개의 head로 나뉨. 각각 768/12=64차원

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=block.attn_dropout.p if block.training else 0.0,
                    is_causal=True,
                ) # 어텐션 스코어 행렬 계산

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(batch_size, i, 768)
                # 12개 head의 출력들 다시 합치기 

                attn_output = block.attn.c_proj(attn_output) # 여기가 W_O
                attn_output = block.attn.resid_dropout(attn_output)

                hidden_states = attn_output + residual
                # Self-Attention 끝
                # Feed-Forward Network 시작
                residual = hidden_states
                hidden_states = block.ln_2(hidden_states) # layer normalization

                feed_forward_hidden_states = block.mlp.c_fc(hidden_states) # W_fc1: 768차원에서 3072차원으로
                feed_forward_hidden_states = block.mlp.act(feed_forward_hidden_states) # 활성화함수 GELU
                feed_forward_hidden_states = block.mlp.c_proj(feed_forward_hidden_states) # W_fc2: 3072차원에서 768차원으로 
                feed_forward_hidden_states = block.mlp.dropout(feed_forward_hidden_states)

                hidden_states = residual + feed_forward_hidden_states
                # Feed-Forward Network 끝

            transformer_outputs = model.transformer.ln_f(hidden_states)
            transformer_outputs = transformer_outputs.view((-1, i, 768))

            outputs = model.lm_head(transformer_outputs)

            next_token_logits = outputs[:, -1, :].clone().float().to(device)

            for b in range(batch_size):
                if labels[b, i] != -100:
                    loss += torch.nn.functional.cross_entropy(next_token_logits[b], labels[b, i])

        loss /= valid_indices.sum().item()
        print(loss)
    
    return (0, 0)