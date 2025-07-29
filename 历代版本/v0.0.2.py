import os
import torch
import torch.nn as nn

printx = print
def print(*args, **kwargs):
    # 绿色
    printx(f'\033[92m{" ".join(map(str, args))}\033[0m', **kwargs)


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_vocab()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.embedding_dim = 256
        self.hidden_dim = 2048
        self.max_length = 128
        self.hidden_state = None
        self.attention_heads = 4
        self.cache = None  # 添加缓存用于生成时的状态复用
    
    def new(self, embedding_dim=256, hidden_dim=2048, max_length=128, attention_heads=4):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.attention_heads = attention_heads
        self._init_model()
        print(f"[模型] 初始化完成，词汇表大小: {len(self.vocab)}，注意力头数: {self.attention_heads}")
    
    def _init_vocab(self):
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<S>']
        ascii_chars = [chr(i) for i in range(32, 127)]
        chinese_punctuation = '，。！？；："“”‘’（）【】《》、'
        common_chinese = [chr(i) for i in range(0x4E00, 0x9FA5 + 1)]
        
        self.vocab = (
            special_tokens + 
            [' ', '\n', '\t'] + 
            ascii_chars + 
            list(chinese_punctuation) + 
            common_chinese
        )
        
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        print(f"[启动] 内存字符集初始化完成，词汇表大小: {len(self.vocab)}")
    
    def _init_model(self):
        if self.vocab is None:
            raise ValueError("词汇表未初始化")
        self.model = TextGeneratorModel(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.attention_heads
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.char2idx.get('<PAD>', 0)).to(self.device)
        self.reset_state()
    
    def reset_state(self):
        self.hidden_state = None
        self.cache = None  # 重置缓存
    
    def _encode_text(self, text):
        encoded = []
        i = 0
        n = len(text)
        
        while i < n:
            if text.startswith('<BOS>', i):
                encoded.append(self.char2idx['<BOS>'])
                i += 5
            elif text.startswith('<EOS>', i):
                encoded.append(self.char2idx['<EOS>'])
                i += 5
            elif text.startswith('<S>', i):
                encoded.append(self.char2idx['<S>'])
                i += 3
            elif text.startswith('<PAD>', i):
                encoded.append(self.char2idx['<PAD>'])
                i += 5
            elif text.startswith('<UNK>', i):
                encoded.append(self.char2idx['<UNK>'])
                i += 5
            else:
                char = text[i]
                encoded.append(self.char2idx.get(char, self.char2idx['<UNK>']))
                i += 1
        
        return encoded
    
    def train(self, prompt, target, study_lr=0.001, epochs=1):
        if self.model is None:
            raise ValueError("模型未初始化")
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = study_lr
        
        input_seq = self._encode_text(prompt)
        target_seq = self._encode_text(target)
        
        # 完整输入序列 = prompt + target[:-1]
        full_input = input_seq + target_seq[:-1]
        
        # 确保序列长度不超过最大长度
        if len(full_input) > self.max_length:
            full_input = full_input[-self.max_length:]
        
        # 输入序列长度
        input_length = len(full_input)
        
        # 目标序列长度
        target_length = len(target_seq)
        
        # 检查长度是否匹配
        if input_length != target_length:
            target_seq = target_seq[:input_length]
            target_length = len(target_seq)
        
        # 转换为张量
        inputs = torch.tensor([full_input], dtype=torch.long).to(self.device)
        targets = torch.tensor([target_seq], dtype=torch.long).to(self.device)
        
        # 1. 创建填充掩码（Padding Mask）
        pad_mask = (inputs != self.char2idx['<PAD>']).unsqueeze(1).unsqueeze(1).float().to(self.device)
        
        # 2. 创建因果掩码（Causal Mask）
        causal_mask = torch.tril(torch.ones((1, 1, input_length, input_length), device=self.device))
        
        # 3. 组合掩码（Padding Mask + Causal Mask）
        mask = pad_mask * causal_mask
        
        # 训练循环
        self.model.train()
        last_loss = 0.0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs, hidden_state, _ = self.model(inputs, mask=mask)
            
            # 确保输出和目标形状匹配
            if outputs.shape[1] != targets.shape[1]:
                outputs = outputs[:, :targets.shape[1], :]
            
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
            last_loss = loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Epoch {epoch + 1}/{epochs} Loss: {last_loss:.4f}")
        
        return last_loss
    
    def train_batch(self, samples, study_lr=0.001, epochs=1, batch_size=32):
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = study_lr
        
        # 1. 数据预处理（保持在 CPU 上，便于 pin_memory 工作）
        input_seqs = []
        target_seqs = []
        pad_idx = self.char2idx['<PAD>']  # PAD 索引
        
        for prompt, target in samples:
            # 编码并截断
            input_seq = self._encode_text(prompt)
            target_seq = self._encode_text(target)
            full_input = input_seq + target_seq[:-1]
            
            # 截断到最大长度
            if len(full_input) > self.max_length:
                full_input = full_input[-self.max_length:]
                target_seq = target_seq[-len(full_input):]
            
            # 对齐长度
            min_len = min(len(full_input), len(target_seq))
            full_input = full_input[:min_len]
            target_seq = target_seq[:min_len]
            
            # 保持在 CPU 上
            input_seqs.append(torch.tensor(full_input, dtype=torch.long))
            target_seqs.append(torch.tensor(target_seq, dtype=torch.long))
        
        # 2. 批量填充（CPU 上操作）
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_seqs, batch_first=True, padding_value=pad_idx
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=pad_idx
        )
        
        # 3. DataLoader 处理 CPU 张量
        dataset = torch.utils.data.TensorDataset(padded_inputs, padded_targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        
        # 训练循环
        self.model.train()
        total_loss = 0.0
        device = self.device
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_inputs, batch_targets in dataloader:
                # 移到 GPU
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                # 获取序列长度
                seq_len = batch_inputs.size(1)  # 添加这行
                
                # 创建填充掩码
                pad_mask = (batch_inputs != pad_idx).unsqueeze(1).unsqueeze(1).float().to(device)
                # 创建因果掩码
                causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device))
                # 组合掩码
                mask = pad_mask * causal_mask
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs, hidden_state, _ = self.model(batch_inputs, mask=mask)
                
                # 计算损失
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    batch_targets.reshape(-1)
                )
                
                # 反向传播
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 输出进度
                if batch_count % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} Batch {batch_count}/{len(dataloader)} Loss: {batch_loss:.4f}")
            
            # 计算每个epoch的平均损失
            avg_loss = epoch_loss / batch_count
            total_loss += avg_loss
            
            # 输出每个epoch的最终损失
            print(f"Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.4f}")
        
        return total_loss / epochs

    def sliding_window_train(self, text, study_lr=0.001, epochs=1, batch_size=32, step_ratio=0.5):
        """高效批量训练方法，支持文本列表输入"""
        # 如果输入是单个字符串，转换为列表
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        # 批量处理所有文本
        all_samples = []
        min_valid_length = 2  # 窗口至少需要2个字符才能生成有效样本
        
        for text in texts:
            encoded_text = self._encode_text(text)
            total_length = len(encoded_text)
            
            # 跳过过短的文本
            if total_length < min_valid_length:
                print(f"文本过短（长度{total_length}），跳过处理")
                continue
                
            # 短文本处理
            if total_length <= self.max_length:
                prompt = encoded_text[:-1]  # 前n-1个字符
                target = encoded_text[1:]   # 后n-1个字符
                if len(prompt) > 0 and len(target) > 0:
                    prompt_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in prompt)
                    target_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in target)
                    all_samples.append((prompt_text, target_text))
            else:
                # 长文本滑动窗口处理
                step = max(int(self.max_length * step_ratio), 1)
                start = 0
                while start + min_valid_length <= total_length:
                    end = min(start + self.max_length, total_length)
                    window = encoded_text[start:end]
                    
                    # 确保窗口有效
                    if len(window) < min_valid_length:
                        start += step
                        continue
                        
                    prompt = window[:-1]
                    target = window[1:]
                    
                    if len(prompt) > 0 and len(target) > 0:
                        prompt_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in prompt)
                        target_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in target)
                        all_samples.append((prompt_text, target_text))
                    
                    start += step
                    
                    # 处理最后一个窗口
                    if end == total_length:
                        break
        
        # 检查是否有有效样本
        if not all_samples:
            print("未生成有效样本，跳过训练")
            return 0.0
            
        print(f"生成 {len(all_samples)} 个有效训练样本")
        return self.train_batch(all_samples, study_lr, epochs, batch_size)
    
    def generate(self, prompt, temperature=0.7, length=50):
        if self.model is None:
            raise ValueError("模型未初始化")
            
        self.model.eval()
        input_seq = self._encode_text(prompt)
        generated = []
        self.cache = None  # 重置缓存
        
        with torch.no_grad():
            for i in range(length):
                inputs = torch.tensor([input_seq], dtype=torch.long).to(self.device)
                
                # 创建更智能的掩码
                seq_len = inputs.size(1)
                if self.cache is None:
                    # 初始生成：完整因果掩码
                    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len))).to(self.device)
                else:
                    # 后续生成：仅需关注当前token
                    mask = torch.ones((1, 1, 1, seq_len)).to(self.device)
                
                # 前向传播，使用缓存
                outputs, self.hidden_state, self.cache = self.model(
                    inputs, 
                    hidden_state=self.hidden_state,
                    cache=self.cache,
                    mask=mask
                )
                
                last_output = outputs[0, -1, :]
                
                # === 关键修改：增加重复惩罚 ===
                if generated:
                    # 对最近出现的token应用惩罚
                    recent_tokens = generated[-min(10, len(generated)):]
                    penalty = torch.ones_like(last_output)
                    for token in recent_tokens:
                        penalty[token] *= 0.7  # 惩罚因子
                    last_output = last_output * penalty
                
                if temperature > 0:
                    probs = torch.softmax(last_output / temperature, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()
                else:
                    next_char_idx = torch.argmax(last_output).item()
                
                # 检查结束符
                char = self.idx2char.get(next_char_idx, '<UNK>')
                if char == '<EOS>':
                    break
                    
                generated.append(next_char_idx)
                input_seq.append(next_char_idx)
                
                # 避免无限循环的额外检查
                if len(generated) > 10 and len(set(generated[-10:])) < 3:
                    # 检测到重复模式，提前终止
                    break
                    
                if len(input_seq) > self.max_length:
                    input_seq = input_seq[-self.max_length:]
                    self.cache = None
        
        chars = [self.idx2char.get(idx, '<UNK>') for idx in generated]
        return ''.join(chars)
    
    def save(self, save_path):
        if self.model is None or self.vocab is None:
            raise ValueError("模型或词汇表未初始化")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_length': self.max_length,
            'attention_heads': self.attention_heads
        }, save_path)
        print(f"[模型] 已保存到 {save_path}")
    
    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件 {load_path} 不存在")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.char2idx = checkpoint['char2idx']
        self.idx2char = checkpoint['idx2char']
        self.embedding_dim = checkpoint.get('embedding_dim', 256)
        self.hidden_dim = checkpoint.get('hidden_dim', 2048)
        self.max_length = checkpoint.get('max_length', 1024)
        self.attention_heads = checkpoint.get('attention_heads', 4)
        
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.reset_state()
        print(f"[模型] 已从 {load_path} 加载，注意力头数: {self.attention_heads}")

class SelfAttention(nn.Module):
    """自注意力机制层（带因果掩码）"""
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "嵌入大小需要能被头数整除"
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask=None, cache=None):
        # 获取批量大小
        N = query.shape[0]
        
        # 处理缓存机制
        if cache is not None:
            # 如果有缓存，只使用当前token
            keys = torch.cat([cache["prev_keys"], keys], dim=1)
            values = torch.cat([cache["prev_values"], values], dim=1)
            query = query[:, -1:, :]  # 只关注最后一个token
        
        # 更新缓存
        new_cache = {
            "prev_keys": keys,
            "prev_values": values
        }
        
        # 获取当前序列长度
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 线性变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # 分割多头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        
        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # 应用掩码（如果有）
        if mask is not None:
            # 确保掩码是4维张量 [batch, heads, query_len, key_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
                
            # 调整掩码尺寸以匹配energy
            if mask.size(-1) != key_len or mask.size(-2) != query_len:
                # 创建正确尺寸的掩码
                new_mask = torch.ones(
                    (N, self.heads, query_len, key_len), 
                    device=energy.device
                )
                # 对于生成模式（cache存在），只需要关注最后一个token
                if cache is not None:
                    new_mask[:, :, -1:, :] = 1
                else:
                    # 训练模式使用下三角掩码
                    new_mask = torch.tril(new_mask)
            else:
                new_mask = mask
            
            energy = energy.masked_fill(new_mask == 0, float("-1e20"))
        
        # 计算注意力权重
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # 应用注意力权重到值上
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        # 通过最终线性层
        out = self.fc_out(out)
        return out, new_cache

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=2048, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )
        
        # 自注意力层
        self.attention = SelfAttention(hidden_dim, num_heads)  # 注意：输入维度改为hidden_dim
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络 - 修正维度匹配问题
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self._init_weights()

        # 增加层归一化
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 增加残差连接
        self.residual_factor = 0.5
    
    def _init_weights(self):
        # 更全面的权重初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                elif 'attention' in name or 'fc' in name or 'output_layer' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # 初始化嵌入层
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
    
    def forward(self, x, hidden_state=None, cache=None, mask=None):
        # 嵌入层
        emb = self.embedding(x)
        
        # LSTM层
        if hidden_state is not None:
            lstm_out, hidden_state = self.lstm(emb, hidden_state)
        else:
            lstm_out, hidden_state = self.lstm(emb)
        
        # 自注意力层 - 使用LSTM的输出作为输入
        attn_out, new_cache = self.attention(
            values=lstm_out,
            keys=lstm_out,
            query=lstm_out,
            mask=mask,
            cache=cache
        )
        
        # 增强残差连接
        lstm_out = self.norm1(lstm_out + self.residual_factor * attn_out)
        
        # 前馈网络
        ff_out = self.fc(lstm_out)
        
        # 增强残差连接
        lstm_out = self.norm2(lstm_out + self.residual_factor * ff_out)
        
        # 新增层归一化
        lstm_out = self.norm3(lstm_out)
        
        # 输出层
        output = self.output_layer(lstm_out)
        
        return output, hidden_state, new_cache