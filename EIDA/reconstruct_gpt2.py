import os
import shutil
import torch
import random


def forward_gpt2(model, dataset_input_ids, dataset_labels, begin, end, batch_size, max_length, p):
    # args:
    # model: 허깅페이스의 "gpt2" 모델
    # dataset_input_ids, dataset_attention_mask: tokenize된 데이터셋 전체
    # begin, end: 데이터셋에서 표본추출에 이용할 구간을 지정
    # p: 표본추출 공간에 들어온 각각의 토큰을 표본으로 뽑을 확률(당연히 PAD토큰은 추출에서 제외함)

    device = model.device

    sample_inputs = [[] for _ in range(12*4)]
    # 파라미터(c_attn, W_O, W_fc1, W_fc2)들의 input token representation이 머무는 latent space들을 지칭하는 index:
    # 4*l+0: block[l]에서 c_attn의 input의 위치 (768차원) (l = 0, 1, ..., 11)
    # 4*l+1: block[l]에서 W_O의 input의 위치 (768차원)
    # 4*l+2: block[l]에서 W_fc1의 input의 위치 (768차원)
    # 4*l+3: block[l]에서 W_fc2의 input의 위치 (3072차원)

    sample_delta_outputs = [[] for _ in range(12*6)]
    # 파라미터들의 output token representation이 머무는 latent space들을 지칭하는 index:
    # 6*l+0: block[l]에서 W_Q의 output(c_attn의 output의 차원들 중 앞 1/3)인 hidden states가 위치하는 곳 (768차원)
    # 6*l+1: block[l]에서 W_K의 output(c_attn의 output의 차원들 중 중간 1/3)인 hidden states가 위치하는 곳 (768차원)
    # 6*l+2: block[l]에서 W_V의 output(c_attn의 output의 차원들 중 뒤 1/3)인 hidden states가 위치하는 곳 (768차원)
    # 6*l+3: block[l]에서 W_O의 output인 hidden states가 위치하는 곳 (768차원)
    # 6*l+4: block[l]에서 W_fc1의 output인 hidden states가 위치하는 곳 (3072차원)
    # 6*l+5: block[l]에서 W_fc2의 output인 hidden states가 위치하는 곳 (768차원)
    # 6*12+0: classifier의 첫번째 계층(768x768)의 output인 hidden states가 위치하는 곳 (768차원)


    position_ids = torch.arange(0, max_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, max_length) # 각 시퀀스별로, 0에서 시작해서 1씩 증가하는 배열. 모델에 input_ids 넣어줄 때 같이 들어감

    for k in range(begin, end, batch_size): # 데이터셋의 [begin:end] 범위를 batch_size만큼씩 잘라서 모델에 집어넣는 루프
        input_ids = dataset_input_ids[k:k+batch_size].clone().to(device)
        labels = dataset_labels[k:k+batch_size].clone().to(device)
        
        valid_indices = labels != -100 # input sequence는 context + completion + padding 으로 구성되어 있는데, 모델은 context를 입력받아 completion을 생성하는 태스크를 수행한다. 입력 데이터에서 completion 부분을 제외한 곳의 label은 -100으로 전처리되어있다.

        sample_size_memo=[] # input 표본들이 담긴 list에서 이번 루프에서는 output 표본 계산에 사용할 범위가 어디부터인지 표시하는 용도
        for idx_param in range(4*12):
            sample_size_memo.append(len(sample_inputs[idx_param]))

        loss = torch.tensor(0.0, device=device)
        
        
        for i in range(1, max_length): # text-generation을 수행하는 auto-regressive decoding loop
            if labels[:, i].sum() == -100 * batch_size: # 배치 안의 모든 시퀀스 중 i번째 토큰이 completion 영역에 있는 경우가 하나도 없으면 i번째 루프는 생략함
                continue

            causal_input_ids = input_ids[:, 0:i].clone().to(device)
            causal_position_ids = position_ids[:, 0:i].clone().to(device)
            # 이 과정에서 KV caching은 이용하지 않음

            inputs_embeds = model.transformer.wte(causal_input_ids) # Word Token Embeddings: vocab_sz=50257 가지의 토큰을 d_model=768 차원 벡터로 변환
            position_embeds = model.transformer.wpe(causal_position_ids) # 시퀀스 내 토큰에 위치정보 입히기
            hidden_states = inputs_embeds + position_embeds
            hidden_states = model.transformer.drop(hidden_states)

            # decoder 부분: block 12개
            for l in range(len(model.transformer.h)):
                block = model.transformer.h[l]

                # Self-Attention 시작
                residual = hidden_states
                hidden_states = block.ln_1(hidden_states) # layer normalization
                
                # W_Q, W_K, W_V의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            sample_inputs[4*l+0].append(hidden_states[b,j,:].clone())
                            # sample_inputs의 인덱스 4*l+0: block[l]의 W_Q, W_K, W_V의 input token representation

                # W_Q, W_K, W_V의 input token representation에 W_Q, W_K, W_V 작용
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
                    dropout_p=block.attn.attn_dropout.p if block.training else 0.0,
                    is_causal=True,
                ) # 어텐션 스코어 행렬 계산

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(batch_size, i, 768)
                # 12개 head의 출력들 다시 합치기 

                # W_O의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            sample_inputs[4*l+1].append(attn_output[b,j,:].clone())
                            # sample_inputs의 인덱스 4*l+1: block[l]의 W_O의 input token representation

                attn_output = block.attn.c_proj(attn_output) # 여기가 W_O
                attn_output = block.attn.resid_dropout(attn_output)

                hidden_states = attn_output + residual
                # Self-Attention 끝
                # Feed-Forward Network 시작
                residual = hidden_states
                hidden_states = block.ln_2(hidden_states) # layer normalization

                # W_fc1의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            sample_inputs[4*l+2].append(hidden_states[b,j,:].clone())
                            # sample_inputs의 인덱스 4*l+2: block[l]의 W_fc1의 input token representation

                feed_forward_hidden_states = block.mlp.c_fc(hidden_states) # W_fc1: 768차원에서 3072차원으로
                feed_forward_hidden_states = block.mlp.act(feed_forward_hidden_states) # 활성화함수 GELU

                # W_fc2의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            sample_inputs[4*l+3].append(feed_forward_hidden_states[b,j,:].clone())
                            # sample_inputs의 인덱스 4*l+3: block[l]의 W_fc2의 input token representation

                feed_forward_hidden_states = block.mlp.c_proj(feed_forward_hidden_states) # W_fc2: 3072차원에서 768차원으로 
                feed_forward_hidden_states = block.mlp.dropout(feed_forward_hidden_states)

                hidden_states = residual + feed_forward_hidden_states
                # Feed-Forward Network 끝

            transformer_outputs = model.transformer.ln_f(hidden_states) # 마지막 layer normalization
            transformer_outputs = transformer_outputs.view((-1, i, 768))

            outputs = model.lm_head(transformer_outputs) # 이 마지막 lm_head는 맨 처음 wte의 transpose임
            # GPT2 모델 끝
            next_token_logits = outputs[:, -1, :].clone().float().to(device)

            for b in range(batch_size):
                if labels[b, i] != -100:
                    loss += torch.nn.functional.cross_entropy(next_token_logits[b], labels[b, i])

        loss /= valid_indices.sum().item() # valid_indices.sum()는 labels가 -100으로 처리되지 않은 자리의 수로, 유효한 logits, labels 비교가 몇 번 일어났는지를 의미. loss는 이 횟수에 대한 평균값 
        loss.backward() # 파라미터 W들의 gradient ΔW 산출.

        # 각 가중치의 Δoutput token representation 계산:
        # 파라미터 W의 input token representation: X, output token representation: Y (Y = WX, Y+ΔY = (W+ΔW)X)
        # 추출된 X에 각 파라미터의 gradient ΔW를 곱하여 ΔY를 계산하는 과정
        with torch.no_grad():
            for l in range(len(model.transformer.h)):
                block = model.transformer.h[l]
                for idx_repres in range(sample_size_memo[4*l+0], len(sample_inputs[4*l+0])):
                    token_delta_output = sample_inputs[4*l+0][idx_repres] @ block.attn.c_attn.weight.grad
                    sample_delta_outputs[6*l+0].append(token_delta_output[0:768].clone())
                    # sample_delta_outputs의 인덱스 6*l+0: block[l]의 W_Q의 output token representation
                    sample_delta_outputs[6*l+1].append(token_delta_output[768:1536].clone())
                    # sample_delta_outputs의 인덱스 6*l+1: block[l]의 W_K의 output token representation
                    sample_delta_outputs[6*l+2].append(token_delta_output[1536:2304].clone())
                    # sample_delta_outputs의 인덱스 6*l+2: block[l]의 W_V의 output token representation
                for idx_repres in range(sample_size_memo[4*l+1], len(sample_inputs[4*l+1])):
                    sample_delta_outputs[6*l+3].append(sample_inputs[4*l+1][idx_repres] @ block.attn.c_proj.weight.grad)
                    # sample_delta_outputs의 인덱스 6*l+3: block[l]의 W_O의 output token representation
                for idx_repres in range(sample_size_memo[4*l+2], len(sample_inputs[4*l+2])):
                    sample_delta_outputs[6*l+4].append(sample_inputs[4*l+2][idx_repres] @ block.mlp.c_fc.weight.grad)
                    # sample_delta_outputs의 인덱스 6*l+4: block[l]의 W_fc1의 output token representation
                for idx_repres in range(sample_size_memo[4*l+3], len(sample_inputs[4*l+3])):
                    sample_delta_outputs[6*l+5].append(sample_inputs[4*l+3][idx_repres] @ block.mlp.c_proj.weight.grad)
                    # sample_delta_outputs의 인덱스 6*l+5: block[l]의 W_fc2의 output token representation

        optim = torch.optim.SGD(model.parameters())
        optim.zero_grad(set_to_none=True)
        del loss, input_ids, labels, valid_indices, causal_input_ids, causal_position_ids, inputs_embeds, position_embeds, hidden_states, query, key, value, attn_output, residual, feed_forward_hidden_states, transformer_outputs, outputs, next_token_logits
        torch.cuda.empty_cache()
        
    return (sample_inputs, sample_delta_outputs) # 두 list 반환



def forward_gpt2_with_save(model, dataset_input_ids, dataset_labels, begin, end, batch_size, max_length, p, dir):
    # args:
    # model: 허깅페이스의 "gpt2" 모델
    # dataset_input_ids, dataset_attention_mask: tokenize된 데이터셋 전체
    # begin, end: 데이터셋에서 표본추출에 이용할 구간을 지정
    # p: 표본추출 공간에 들어온 각각의 토큰을 표본으로 뽑을 확률(당연히 PAD토큰은 추출에서 제외함)

    device = model.device

    if os.path.exists(dir):
        shutil.rmtree(dir) # 지정된 경로에 파일이 있으면 다 지우기
        
    for i in range(4*12):
        os.makedirs(os.path.join(dir, "inputs", f"{i}"), exist_ok=True)
    # 파라미터(c_attn, W_O, W_fc1, W_fc2)들의 input token representation이 머무는 latent space들을 지칭하는 index:
    # 4*l+0: block[l]에서 c_attn의 input의 위치 (768차원) (l = 0, 1, ..., 11)
    # 4*l+1: block[l]에서 W_O의 input의 위치 (768차원)
    # 4*l+2: block[l]에서 W_fc1의 input의 위치 (768차원)
    # 4*l+3: block[l]에서 W_fc2의 input의 위치 (3072차원)

    for i in range(6*12):
        os.makedirs(os.path.join(dir, "delta_outputs", f"{i}"), exist_ok=True)
    # 파라미터들의 output token representation이 머무는 latent space들을 지칭하는 index:
    # 6*l+0: block[l]에서 W_Q의 output(c_attn의 output의 차원들 중 앞 1/3)인 hidden states가 위치하는 곳 (768차원)
    # 6*l+1: block[l]에서 W_K의 output(c_attn의 output의 차원들 중 중간 1/3)인 hidden states가 위치하는 곳 (768차원)
    # 6*l+2: block[l]에서 W_V의 output(c_attn의 output의 차원들 중 뒤 1/3)인 hidden states가 위치하는 곳 (768차원)
    # 6*l+3: block[l]에서 W_O의 output인 hidden states가 위치하는 곳 (768차원)
    # 6*l+4: block[l]에서 W_fc1의 output인 hidden states가 위치하는 곳 (3072차원)
    # 6*l+5: block[l]에서 W_fc2의 output인 hidden states가 위치하는 곳 (768차원)
    # 6*12+0: classifier의 첫번째 계층(768x768)의 output인 hidden states가 위치하는 곳 (768차원)

    sample_counter_write = [0 for _ in range(4*12)]
    sample_counter_read = [0 for _ in range(4*12)]

    position_ids = torch.arange(0, max_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, max_length) # 각 시퀀스별로, 0에서 시작해서 1씩 증가하는 배열. 모델에 input_ids 넣어줄 때 같이 들어감

    for k in range(begin, end, batch_size): # 데이터셋의 [begin:end] 범위를 batch_size만큼씩 잘라서 모델에 집어넣는 루프
        input_ids = dataset_input_ids[k:k+batch_size].clone().to(device)
        labels = dataset_labels[k:k+batch_size].clone().to(device)
        
        valid_indices = labels != -100 # input sequence는 context + completion + padding 으로 구성되어 있는데, 모델은 context를 입력받아 completion을 생성하는 태스크를 수행한다. 입력 데이터에서 completion 부분을 제외한 곳의 label은 -100으로 전처리되어있다.

        loss = torch.tensor(0.0, device=device)

        for i in range(1, max_length): # text-generation을 수행하는 auto-regressive decoding loop
            if labels[:, i].sum() == -100 * batch_size: # 배치 안의 모든 시퀀스 중 i번째 토큰이 completion 영역에 있는 경우가 하나도 없으면 i번째 루프는 생략함
                continue

            causal_input_ids = input_ids[:, 0:i].clone().to(device)
            causal_position_ids = position_ids[:, 0:i].clone().to(device)
            # 이 과정에서 KV caching은 이용하지 않음

            inputs_embeds = model.transformer.wte(causal_input_ids) # Word Token Embeddings: vocab_sz=50257 가지의 토큰을 d_model=768 차원 벡터로 변환
            position_embeds = model.transformer.wpe(causal_position_ids) # 시퀀스 내 토큰들에 위치정보 입히기
            hidden_states = inputs_embeds + position_embeds
            hidden_states = model.transformer.drop(hidden_states)

            # decoder 부분: block 12개개
            for l in range(len(model.transformer.h)):
                block = model.transformer.h[l]

                # Self-Attention 시작
                residual = hidden_states
                hidden_states = block.ln_1(hidden_states) # layer normalization
                
                # W_Q, W_K, W_V의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            torch.save(hidden_states[b,j,:].clone(), os.path.join(dir, "inputs", f"{4*l+0}", f"{sample_counter_write[4*l+0]}.pt"))
                            sample_counter_write[4*l+0] += 1
                            # sample_inputs의 인덱스 4*l+0: block[l]의 W_Q, W_K, W_V의 input token representation

                # W_Q, W_K, W_V의 input token representation에 W_Q, W_K, W_V 작용
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

                # W_O의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            torch.save(attn_output[b,j,:].clone(), os.path.join(dir, "inputs", f"{4*l+1}", f"{sample_counter_write[4*l+1]}.pt"))
                            sample_counter_write[4*l+1] += 1
                            # sample_inputs의 인덱스 4*l+1: block[l]의 W_O의 input token representation

                attn_output = block.attn.c_proj(attn_output) # 여기가 W_O
                attn_output = block.attn.resid_dropout(attn_output)

                hidden_states = attn_output + residual
                # Self-Attention 끝
                # Feed-Forward Network 시작
                residual = hidden_states
                hidden_states = block.ln_2(hidden_states) # layer normalization

                # W_fc1의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            torch.save(hidden_states[b,j,:].clone(), os.path.join(dir, "inputs", f"{4*l+2}", f"{sample_counter_write[4*l+2]}.pt"))
                            sample_counter_write[4*l+2] += 1
                            # sample_inputs의 인덱스 4*l+2: block[l]의 W_fc1의 input token representation

                feed_forward_hidden_states = block.mlp.c_fc(hidden_states) # W_fc1: 768차원에서 3072차원으로
                feed_forward_hidden_states = block.mlp.act(feed_forward_hidden_states) # 활성화함수 GELU

                # W_fc2의 input token representation 표본 추출
                for b in range(batch_size):
                    for j in range(i):
                        if random.random() < p:
                            torch.save(feed_forward_hidden_states[b,j,:].clone(), os.path.join(dir, "inputs", f"{4*l+3}", f"{sample_counter_write[4*l+3]}.pt"))
                            sample_counter_write[4*l+3] += 1
                            # sample_inputs의 인덱스 4*l+3: block[l]의 W_fc2의 input token representation

                feed_forward_hidden_states = block.mlp.c_proj(feed_forward_hidden_states) # W_fc2: 3072차원에서 768차원으로 
                feed_forward_hidden_states = block.mlp.dropout(feed_forward_hidden_states)

                hidden_states = residual + feed_forward_hidden_states
                # Feed-Forward Network 끝

            transformer_outputs = model.transformer.ln_f(hidden_states)
            transformer_outputs = transformer_outputs.view((-1, i, 768))

            outputs = model.lm_head(transformer_outputs) # 이 마지막 lm_head는 맨 처음 wte의 transpose임
            # GPT2 모델 끝끝
            next_token_logits = outputs[:, -1, :].clone().float().to(device)

            for b in range(batch_size):
                if labels[b, i] != -100:
                    loss += torch.nn.functional.cross_entropy(next_token_logits[b], labels[b, i])

        loss /= valid_indices.sum().item() # valid_indices.sum()는 labels가 -100으로 처리되지 않은 자리의 수로, 유효한 logits, labels 비교가 몇 번 일어났는지를 의미. loss는 이 횟수에 대한 평균값

        optim = torch.optim.SGD(model.parameters()) # 옵티마이저 쓸려고 만든 게 아니고 뒤에서 optim.zero_grad() 쓸려고 만듬

        loss.backward() # 파라미터 W들의 gradient ΔW 산출.

        # 각 가중치의 Δoutput token representation 계산:
        # 파라미터 W의 input token representation: X, output token representation: Y (Y = WX, Y+ΔY = (W+ΔW)X)
        # 추출된 X에 각 파라미터의 gradient ΔW를 곱하여 ΔY를 계산하는 과정
        for l in range(len(model.transformer.h)):
            block = model.transformer.h[l]
            for idx_repres in range(sample_counter_read[4*l+0], sample_counter_write[4*l+0]):
                loaded_token = torch.load(os.path.join(dir, "inputs", f"{4*l+0}", f"{idx_repres}.pt"), weights_only=True)
                sample_counter_read[4*l+0] += 1
                token_delta_output = loaded_token @ block.attn.c_attn.weight.grad
                torch.save(token_delta_output[0:768].clone(), os.path.join(dir, "delta_outputs", f"{6*l+0}", f"{idx_repres}.pt"))
                # sample_delta_outputs의 인덱스 6*l+0: block[l]의 W_Q의 output token representation
                torch.save(token_delta_output[768:1536].clone(), os.path.join(dir, "delta_outputs", f"{6*l+1}", f"{idx_repres}.pt"))
                # sample_delta_outputs의 인덱스 6*l+1: block[l]의 W_K의 output token representation
                torch.save(token_delta_output[1536:2304].clone(), os.path.join(dir, "delta_outputs", f"{6*l+2}", f"{idx_repres}.pt"))
                # sample_delta_outputs의 인덱스 6*l+2: block[l]의 W_V의 output token representation
            for idx_repres in range(sample_counter_read[4*l+1], sample_counter_write[4*l+1]):
                loaded_token = torch.load(os.path.join(dir, "inputs", f"{4*l+1}", f"{idx_repres}.pt"), weights_only=True)
                sample_counter_read[4*l+1] += 1
                torch.save(loaded_token @ block.attn.c_proj.weight.grad, os.path.join(dir, "delta_outputs", f"{6*l+3}", f"{idx_repres}.pt"))
                # sample_delta_outputs의 인덱스 6*l+3: block[l]의 W_O의 output token representation
            for idx_repres in range(sample_counter_read[4*l+2], sample_counter_write[4*l+2]):
                loaded_token = torch.load(os.path.join(dir, "inputs", f"{4*l+2}", f"{idx_repres}.pt"), weights_only=True)
                sample_counter_read[4*l+2] += 1
                torch.save(loaded_token @ block.mlp.c_fc.weight.grad, os.path.join(dir, "delta_outputs", f"{6*l+4}", f"{idx_repres}.pt"))
                # sample_delta_outputs의 인덱스 6*l+4: block[l]의 W_fc1의 output token representation
            for idx_repres in range(sample_counter_read[4*l+3], sample_counter_write[4*l+3]):
                loaded_token = torch.load(os.path.join(dir, "inputs", f"{4*l+3}", f"{idx_repres}.pt"), weights_only=True)
                sample_counter_read[4*l+3] += 1
                torch.save(loaded_token @ block.mlp.c_proj.weight.grad, os.path.join(dir, "delta_outputs", f"{6*l+5}", f"{idx_repres}.pt"))
                # sample_delta_outputs의 인덱스 6*l+5: block[l]의 W_fc2의 output token representation

        optim.zero_grad()