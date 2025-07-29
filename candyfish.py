import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 自定义打印函数，使输出更易读
printx = print
def print(*args, **kwargs):
    """自定义打印函数，输出绿色文本"""
    printx(f'\033[92m{" ".join(map(str, args))}\033[0m', **kwargs)

class TextGeneratorModel(nn.Module):
    """文本生成模型，结合LSTM和自注意力机制"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=2048, num_heads=4):
        """
        初始化模型
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层，处理序列信息
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=2,  # 两层LSTM
            batch_first=True,
            bidirectional=False
        )
        
        # 自注意力层，捕捉全局依赖
        self.attention = SelfAttention(hidden_dim, num_heads)
        
        # 层归一化，稳定训练
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈神经网络
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # 扩展维度
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim * 4, hidden_dim)   # 压缩回原始维度
        )
        
        # 输出层，预测下一个token
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重初始化
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                elif 'attention' in name or 'fc' in name or 'output_layer' in name:
                    # 线性层权重初始化
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 偏置初始化
                nn.init.constant_(param, 0.0)
        # 嵌入层初始化
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
    
    def forward(self, x, hidden_state=None, cache=None, mask=None):
        """
        前向传播
        Args:
            x: 输入token序列
            hidden_state: LSTM隐藏状态
            cache: 自注意力缓存
            mask: 注意力掩码
        Returns:
            output: 预测logits
            hidden_state: 更新后的隐藏状态
            new_cache: 更新后的注意力缓存
        """
        # 1. 词嵌入
        emb = self.embedding(x)
        
        # 2. LSTM处理
        if hidden_state is not None:
            lstm_out, hidden_state = self.lstm(emb, hidden_state)
        else:
            lstm_out, hidden_state = self.lstm(emb)
        
        # 3. 自注意力处理
        attn_out, new_cache = self.attention(
            values=lstm_out,
            keys=lstm_out,
            query=lstm_out,
            mask=mask,
            cache=cache
        )
        
        # 4. 残差连接 + 层归一化
        lstm_out = self.norm1(lstm_out + 0.5 * attn_out)
        
        # 5. 前馈网络
        ff_out = self.fc(lstm_out)
        
        # 6. 残差连接 + 层归一化
        lstm_out = self.norm2(lstm_out + 0.5 * ff_out)
        
        # 7. 输出层
        output = self.output_layer(lstm_out)
        
        return output, hidden_state, new_cache

class SelfAttention(nn.Module):
    """自注意力机制层，支持因果掩码和缓存"""
    def __init__(self, embed_size, heads):
        """
        初始化自注意力层
        Args:
            embed_size: 嵌入维度
            heads: 注意力头数
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # 确保维度可分割
        assert self.head_dim * heads == embed_size, "嵌入大小需要能被头数整除"
        
        # 线性变换层
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask=None, cache=None):
        """
        前向传播
        Args:
            values: 值向量
            keys: 键向量
            query: 查询向量
            mask: 注意力掩码
            cache: 缓存的前一状态
        Returns:
            out: 注意力输出
            new_cache: 更新后的缓存
        """
        # 获取批量大小
        N = query.shape[0]
        
        # 处理缓存机制
        if cache is not None:
            # 如果有缓存，拼接当前和之前的键值
            keys = torch.cat([cache["prev_keys"], keys], dim=1)
            values = torch.cat([cache["prev_values"], values], dim=1)
            # 只关注最后一个token的查询
            query = query[:, -1:, :]
        
        # 更新缓存
        new_cache = {
            "prev_keys": keys,
            "prev_values": values
        }
        
        # 获取序列长度
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
            # 调整掩码维度
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
            
            # 应用掩码
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

class Model:
    """文本生成模型封装类"""
    def __init__(self):
        """初始化模型类"""
        # 设备配置（优先使用GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[设备] 使用 {self.device} 进行计算")
        
        # 初始化词汇表
        self._init_vocab()
        
        # 模型组件初始化
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # 模型参数默认值
        self.embedding_dim = 256
        self.hidden_dim = 2048
        self.max_length = 128
        self.attention_heads = 4
        
        # 状态变量
        self.hidden_state = None  # LSTM隐藏状态
        self.cache = None  # 注意力缓存，用于生成时的状态复用
    
    def _init_vocab(self):
        """初始化词汇表，包含特殊字符、ASCII字符和常用汉字"""
        # 特殊token
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<S>']
        
        # ASCII字符（可打印字符）
        ascii_chars = [chr(i) for i in range(32, 127)]
        
        # 中文标点符号
        chinese_punctuation = '，。！？；："“”‘’（）【】《》、'
        
        # 常用汉字（Unicode范围：4E00-9FA5）
        common_chinese = [chr(i) for i in range(0x4E00, 0x9FA5 + 1)]
        
        # 组合词汇表
        self.vocab = (
            special_tokens + 
            [' ', '\n', '\t'] +  # 空白字符
            ascii_chars + 
            list(chinese_punctuation) + 
            common_chinese
        )
        
        # 创建映射字典
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        print(f"[词汇表] 初始化完成，词汇表大小: {len(self.vocab)}")
    
    def new(self, embedding_dim=256, hidden_dim=2048, max_length=128, attention_heads=4):
        """
        创建新模型
        Args:
            embedding_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            max_length: 最大序列长度
            attention_heads: 注意力头数
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.attention_heads = attention_heads
        
        # 初始化模型
        self._init_model()
        print(f"[模型] 初始化完成，参数: "
              f"嵌入={embedding_dim}, 隐藏={hidden_dim}, "
              f"最大长度={max_length}, 注意力头={attention_heads}")
    
    def _init_model(self):
        """初始化模型组件"""
        if self.vocab is None:
            raise ValueError("词汇表未初始化")
        
        # 创建模型实例
        self.model = TextGeneratorModel(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.attention_heads
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # 损失函数（忽略填充token）
        pad_idx = self.char2idx.get('<PAD>', 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(self.device)
        
        # 重置状态
        self.reset_state()
    
    def reset_state(self):
        """重置模型状态（用于新的生成序列）"""
        self.hidden_state = None
        self.cache = None
        print("[状态] 已重置隐藏状态和缓存")
    
    def _encode_text(self, text):
        """
        将文本编码为token索引序列
        Args:
            text: 输入文本
        Returns:
            encoded: token索引列表
        """
        encoded = []
        i = 0
        n = len(text)
        
        # 处理特殊token和普通字符
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
                # 处理单个字符
                char = text[i]
                encoded.append(self.char2idx.get(char, self.char2idx['<UNK>']))
                i += 1
        
        return encoded
    
    def train(self, prompt, study_lr=0.001, epochs=1, batch_size=32):
        """
        统一的训练方法，支持单条和批量训练
        Args:
            prompt: 训练数据，可以是:
                    - 单个元组 (input_str, output_str) 
                    - 元组列表 [(input1, output1), (input2, output2), ...]
            study_lr: 学习率
            epochs: 训练轮数
            batch_size: 批量大小（仅当prompt为列表时有效）
        Returns:
            avg_loss: 平均训练损失
        """
        # 自动检测输入类型
        if isinstance(prompt, tuple) and len(prompt) == 2:
            # 单个样本，转换为列表
            samples = [prompt]
            # 单样本训练时强制batch_size=1
            batch_size = 1  
            print(f"[训练] 单样本训练模式，输入长度={len(prompt[0])}，输出长度={len(prompt[1])}")
        elif isinstance(prompt, list) and all(isinstance(p, tuple) and len(p) == 2 for p in prompt):
            # 批量样本
            samples = prompt
            print(f"[训练] 批量训练模式，样本数={len(samples)}，批量大小={batch_size}")
        else:
            raise ValueError("prompt 必须是 (input, output) 元组或该元组的列表")
        
        # 调用内部批量训练方法
        return self._train_batch(samples, study_lr, epochs, batch_size)
    
    def _train_batch(self, samples, study_lr=0.001, epochs=1, batch_size=32):
        """
        内部批量训练方法
        Args:
            samples: 样本列表 [(input1, output1), ...]
            study_lr: 学习率
            epochs: 训练轮数
            batch_size: 批量大小
        Returns:
            avg_loss: 平均训练损失
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = study_lr
        print(f"[训练] 学习率设置为: {study_lr}")
        
        # 1. 数据预处理（保持在CPU上以减少显存占用）
        input_seqs = []
        target_seqs = []
        pad_idx = self.char2idx['<PAD>']  # 填充token索引
        
        for input_text, output_text in samples:
            # 编码输入和目标文本
            input_seq = self._encode_text(input_text)
            target_seq = self._encode_text(output_text)
            
            # 完整输入序列 = input_text + output_text[:-1]
            # 目标序列 = output_text (与输入序列对齐)
            full_input = input_seq + target_seq[:-1]
            
            # 截断到最大长度
            if len(full_input) > self.max_length:
                full_input = full_input[-self.max_length:]
                target_seq = target_seq[-len(full_input):]
            
            # 对齐长度
            min_len = min(len(full_input), len(target_seq))
            full_input = full_input[:min_len]
            target_seq = target_seq[:min_len]
            
            # 保持在CPU上
            input_seqs.append(torch.tensor(full_input, dtype=torch.long))
            target_seqs.append(torch.tensor(target_seq, dtype=torch.long))
        
        # 2. 批量填充（处理不同长度的序列）
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_seqs, batch_first=True, padding_value=pad_idx
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=pad_idx
        )
        
        # 3. 创建DataLoader
        dataset = TensorDataset(padded_inputs, padded_targets)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True  # 启用快速数据加载
        )
        
        # 训练循环
        self.model.train()
        total_loss = 0.0
        device = self.device
        
        print(f"[训练] 开始训练，总轮数={epochs}，总批次={len(dataloader)}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_inputs, batch_targets in dataloader:
                # 将数据移动到设备（GPU）
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                # 获取序列长度
                seq_len = batch_inputs.size(1)
                
                # 创建填充掩码（忽略填充位置）
                pad_mask = (batch_inputs != pad_idx).unsqueeze(1).unsqueeze(1).float().to(device)
                # 创建因果掩码（防止模型看到未来信息）
                causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device))
                # 组合掩码
                mask = pad_mask * causal_mask
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs, _, _ = self.model(batch_inputs, mask=mask)
                
                # 计算损失（仅计算非填充位置）
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    batch_targets.reshape(-1)
                )
                
                # 反向传播
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # 更新参数
                self.optimizer.step()
                
                # 记录损失
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 每10个批次输出一次进度
                if batch_count % 10 == 0:
                    print(f"轮次 {epoch+1}/{epochs} 批次 {batch_count}/{len(dataloader)} 损失: {batch_loss:.4f}")
            
            # 计算当前轮次的平均损失
            avg_loss = epoch_loss / batch_count
            total_loss += avg_loss
            
            # 输出轮次总结
            print(f"轮次 {epoch+1}/{epochs} 完成，平均损失: {avg_loss:.4f}")
        
        # 计算整体平均损失
        final_avg_loss = total_loss / epochs
        print(f"[训练] 训练完成，最终平均损失: {final_avg_loss:.4f}")
        return final_avg_loss

    def sliding_window_train(self, text, study_lr=0.001, epochs=1, batch_size=32, step_ratio=0.5):
        """
        滑动窗口训练方法，用于长文本
        Args:
            text: 输入文本（字符串或字符串列表）
            study_lr: 学习率
            epochs: 训练轮数
            batch_size: 批量大小
            step_ratio: 滑动窗口步长比例
        Returns:
            avg_loss: 平均训练损失
        """
        # 如果输入是单个字符串，转换为列表
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        # 批量处理所有文本
        all_samples = []
        min_valid_length = 2  # 窗口至少需要2个字符
        
        print(f"[滑动窗口] 处理 {len(texts)} 个文本，窗口大小={self.max_length}，步长比例={step_ratio}")
        
        for text in texts:
            encoded_text = self._encode_text(text)
            total_length = len(encoded_text)
            
            # 跳过过短的文本
            if total_length < min_valid_length:
                print(f"文本过短（长度{total_length}），跳过处理")
                continue
                
            # 短文本处理（直接作为单个样本）
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
            
        print(f"[滑动窗口] 生成 {len(all_samples)} 个训练样本")
        return self.train(all_samples, study_lr, epochs, batch_size)
    
    def generate(self, prompt, temperature=0.7, length=50, repetition_penalty=1.2):
        """
        文本生成方法
        Args:
            prompt: 输入提示文本
            temperature: 温度参数（控制随机性）
            length: 生成的最大长度
            repetition_penalty: 重复惩罚因子
        Returns:
            generated_text: 生成的文本
        """
        if self.model is None:
            raise ValueError("模型未初始化")
            
        # 切换到评估模式
        self.model.eval()
        
        # 重置状态（新的生成序列）
        self.reset_state()
        
        # 编码输入
        input_seq = self._encode_text(prompt)
        generated = []  # 存储生成的token索引
        
        print(f"[生成] 开始生成，初始提示: '{prompt[:30]}...'，温度={temperature}，长度={length}")
        
        with torch.no_grad():
            for i in range(length):
                # 准备输入
                inputs = torch.tensor([input_seq], dtype=torch.long).to(self.device)
                
                # 创建注意力掩码
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
                
                # 获取最后一个token的输出
                last_output = outputs[0, -1, :]
                
                # 重复惩罚（减少重复生成）
                if generated:
                    # 对最近出现的token应用惩罚
                    recent_tokens = generated[-min(10, len(generated)):]
                    penalty = torch.ones_like(last_output)
                    for token in recent_tokens:
                        penalty[token] /= repetition_penalty  # 降低重复token的概率
                    last_output = last_output * penalty
                
                # 采样下一个token
                if temperature > 0:
                    # 使用温度控制的softmax
                    probs = torch.softmax(last_output / temperature, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()
                else:
                    # 贪婪采样（选择概率最高的token）
                    next_char_idx = torch.argmax(last_output).item()
                
                # 检查结束符
                char = self.idx2char.get(next_char_idx, '<UNK>')
                if char == '<EOS>':
                    print(f"[生成] 检测到结束符，提前终止")
                    break
                    
                # 添加到生成序列
                generated.append(next_char_idx)
                input_seq.append(next_char_idx)
                
                # 避免无限循环的额外检查
                if len(generated) > 10 and len(set(generated[-10:])) < 3:
                    print(f"[生成] 检测到重复模式，提前终止")
                    break
                    
                # 限制输入序列长度
                if len(input_seq) > self.max_length:
                    input_seq = input_seq[-self.max_length:]
                    self.cache = None  # 重置缓存
        
        # 将token索引转换为文本
        chars = [self.idx2char.get(idx, '<UNK>') for idx in generated]
        generated_text = ''.join(chars)
        
        print(f"[生成] 完成，生成长度: {len(generated_text)}")
        return generated_text
    
    def save(self, save_path):
        """
        保存模型到文件
        Args:
            save_path: 保存路径
        """
        if self.model is None or self.vocab is None:
            raise ValueError("模型或词汇表未初始化")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存所有必要信息
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
        
        print(f"[保存] 模型已保存到 {save_path}")
    
    def load(self, load_path):
        """
        从文件加载模型
        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件 {load_path} 不存在")
        
        # 加载检查点
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 恢复词汇表
        self.vocab = checkpoint['vocab']
        self.char2idx = checkpoint['char2idx']
        self.idx2char = checkpoint['idx2char']
        
        # 恢复模型参数
        self.embedding_dim = checkpoint.get('embedding_dim', 256)
        self.hidden_dim = checkpoint.get('hidden_dim', 2048)
        self.max_length = checkpoint.get('max_length', 1024)
        self.attention_heads = checkpoint.get('attention_heads', 4)
        
        # 初始化模型
        self._init_model()
        
        # 加载模型和优化器状态
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # 重置状态
        self.reset_state()
        
        print(f"[加载] 模型已从 {load_path} 加载，参数: "
              f"嵌入={self.embedding_dim}, 隐藏={self.hidden_dim}, "
              f"最大长度={self.max_length}, 注意力头={self.attention_heads}")