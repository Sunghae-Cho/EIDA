import torch
import random


def forward(model, dataset_input_ids, dataset_attention_mask, dataset_label, begin, end, batch_size=16, max_length=512, N=2):
    device = model.device

    sample_inputs=[[] for _ in range(12*4+1)]
    sample_delta_outputs=[[] for _ in range(12*6+1)]
    
    for k in range(begin, end, batch_size):
        input_ids = dataset_input_ids[k:k+batch_size].clone().to(device)
        attention_mask = dataset_attention_mask[k:k+batch_size].clone().to(device)
        label = dataset_label[k:k+batch_size].clone().to(device)

        sample_size_memo=[]
        for idx_param in range(4*12+1):
            sample_size_memo.append(len(sample_inputs[idx_param]))

        extended_attention_mask = (1 - attention_mask) * -1e36
        extended_attention_mask = extended_attention_mask[:, None, None, :].expand(batch_size, 1, max_length, max_length)
        extended_attention_mask = extended_attention_mask.to(device, dtype=model.dtype)

        # RoBERTa 모델 시작
        # input embedding 부분
        hidden_states = model.roberta.embeddings(input_ids=input_ids)

        # encoder 부분: 레이어 12개
        for l, layer in enumerate(model.roberta.encoder.layer):
            # 모델의 Self-Attention 부분 시작
            # W_Q, W_K, W_V의 input token representation 표본추출
            for b in range(batch_size):
                random_numbers = random.sample(range(sum(attention_mask[b,:])), N) # 각 시퀀스에서 N개 뽑기
                for i in random_numbers:
                    sample_inputs[4*l+0].append(hidden_states[b,i,:])
                    # sample_inputs의 인덱스 4*l+0: layer-l의 W_Q, W_K, W_V의 input token representation

            # W_Q, W_K, W_V의 input token representation에 W_Q, W_K, W_V 작용
            query_layer = layer.attention.self.query(hidden_states)
            key_layer = layer.attention.self.key(hidden_states)
            value_layer = layer.attention.self.value(hidden_states)
            query_layer = layer.attention.self.transpose_for_scores(query_layer)
            key_layer = layer.attention.self.transpose_for_scores(key_layer)
            value_layer = layer.attention.self.transpose_for_scores(value_layer)

            # (Query와 Key는 12개의 head로 나눠짐) score matrix 계산
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=extended_attention_mask,
                dropout_p=0.1 if layer.attention.training else 0.0,
            )
            
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(batch_size, max_length, 768)

            
            # W_O의 input token representation 표본추출
            for b in range(batch_size):
                random_numbers = random.sample(range(sum(attention_mask[b,:])), N) # 각 시퀀스에서 N개 뽑기
                for i in random_numbers:
                    sample_inputs[4*l+1].append(hidden_states[b,i,:])
                    # sample_inputs의 인덱스 4*l+1: layer-l의 W_O의 input token representation

            # W_O의 input token representation에 W_O를 작용, attention 이전의 입력으로 residual connection, layer normalization 적용
            attention_output = layer.attention.output(attn_output, hidden_states)
            # Self-Attention 부분 끝

            # Feed-Forward Network 부분 시작
            # W_fc1의 input token representation 표본추출
            for b in range(batch_size):
                random_numbers = random.sample(range(sum(attention_mask[b,:])), N) # 각 시퀀스에서 N개 뽑기
                for i in random_numbers:
                    sample_inputs[4*l+2].append(attention_output[b,i,:])
                    # sample_inputs의 인덱스 4*l+2: layer-l의 W_fc1의 input token representation

            # W_fc1의 input token representation에 W_fc1 작용, 활성화함수 GELU 적용
            hidden_states = layer.intermediate.dense(attention_output) # fc1
            hidden_states = layer.intermediate.intermediate_act_fn(hidden_states) # activation function

            # W_fc2의 input token representation 표본추출
            for b in range(batch_size):
                random_numbers = random.sample(range(sum(attention_mask[b,:])), N) # 각 시퀀스에서 N개 뽑기
                for i in random_numbers:
                    sample_inputs[4*l+3].append(hidden_states[b,i,:])
                    # sample_inputs의 인덱스 4*l+3: layer-l의 W_fc2의 input token representation

            # W_fc2의 input token representation에 W_fc2 작용, residual connection, layer normalization 적용
            hidden_states = layer.output.dense(hidden_states) # fc2
            hidden_states = layer.output.dropout(hidden_states) # dropout
            hidden_states = layer.output.LayerNorm(hidden_states + attention_output) # residual connection & layer normalization
            # Feed-Forward Network 부분 끝

        # 모델 끝의 classifier 부분 시작
        # classifier의 input token representation 표본추출
        for b in range(batch_size):
            random_numbers = random.sample(range(sum(attention_mask[b,:])), N) # 각 시퀀스에서 N개 뽑기
            for i in random_numbers:
                sample_inputs[4*12+0].append(hidden_states[b,i,:])
                # sample_inputs의 인덱스 4*12+0: classifier의 input token representation

        x = hidden_states[:, 0, :]  # 시퀀스의 첫 토큰(<s>)을 classifier에 넣음
        x = model.classifier.dropout(x)
        x = model.classifier.dense(x)
        x = torch.tanh(x)

        x = model.classifier.dropout(x)
        logits = model.classifier.out_proj(x)
        # RoBERTa 모델 끝

        loss_fn = torch.nn.CrossEntropyLoss()
        optim = torch.optim.SGD(model.parameters())
        loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))

        loss.backward() # 파라미터 W의 gradient ΔW 산출.

        # 각 가중치의 output token representation 계산:
        # 가중치 W의 input token representation: X, output token representation: Y (Y = WX, Y+ΔY = (W+ΔW)X)
        # 추출된 W의 input token representation X에 각 가중치의 gradient ΔW를 곱하여 ΔY를 계산
        for l, layer in enumerate(model.roberta.encoder.layer):
            for idx_repres in range(sample_size_memo[4*l+0], len(sample_inputs[4*l+0])):
                sample_delta_outputs[6*l+0].append(sample_inputs[4*l+0][idx_repres] @ layer.attention.self.query.weight.grad.T)
                # sample_delta_outputs의 인덱스 6*l+0: layer-l의 W_Q의 output token representation
            for idx_repres in range(sample_size_memo[4*l+0], len(sample_inputs[4*l+0])):
                sample_delta_outputs[6*l+1].append(sample_inputs[4*l+0][idx_repres] @ layer.attention.self.key.weight.grad.T)
                # sample_delta_outputs의 인덱스 6*l+1: layer-l의 W_K의 output token representation
            for idx_repres in range(sample_size_memo[4*l+0], len(sample_inputs[4*l+0])):
                sample_delta_outputs[6*l+2].append(sample_inputs[4*l+0][idx_repres] @ layer.attention.self.value.weight.grad.T)
                # sample_delta_outputs의 인덱스 6*l+2: layer-l의 W_V의 output token representation
            for idx_repres in range(sample_size_memo[4*l+1], len(sample_inputs[4*l+1])):
                sample_delta_outputs[6*l+3].append(sample_inputs[4*l+1][idx_repres] @ layer.attention.output.dense.weight.grad.T)
                # sample_delta_outputs의 인덱스 6*l+3: layer-l의 W_O의 output token representation
            for idx_repres in range(sample_size_memo[4*l+2], len(sample_inputs[4*l+2])):
                sample_delta_outputs[6*l+4].append(sample_inputs[4*l+2][idx_repres] @ layer.intermediate.dense.weight.grad.T)
                # sample_delta_outputs의 인덱스 6*l+4: layer-l의 W_fc1의 output token representation
            for idx_repres in range(sample_size_memo[4*l+3], len(sample_inputs[4*l+3])):
                sample_delta_outputs[6*l+5].append(sample_inputs[4*l+3][idx_repres] @ layer.output.dense.weight.grad.T)
                # sample_delta_outputs의 인덱스 6*l+5: layer-l의 W_fc2의 output token representation
        for idx_repres in range(sample_size_memo[4*12+0], len(sample_inputs[4*12+0])):
            sample_delta_outputs[6*12+0].append(sample_inputs[4*12+0][idx_repres] @ model.classifier.dense.weight.grad.T)
            # sample_delta_outputs의 인덱스 6*12+0: classifier의 첫번째 dense layer(768x768)의 output token representation
            
        optim.zero_grad()
    
    return (sample_inputs, sample_delta_outputs)