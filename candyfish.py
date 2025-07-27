import os
import torch
import torch.nn as nn

printx= print
def print(*args, **kwargs):
    # 绿色
    printx(f'\033[92m{' '.join(map(str, args))}\033[0m', **kwargs)


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_vocab()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.embedding_dim = 512
        self.hidden_dim = 2048
        self.max_length = 128
        self.hidden_state = None
    
    def new(self, embedding_dim=512, hidden_dim=2048, max_length=128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self._init_model()
        print(f"[模型] 初始化完成，词汇表大小: {len(self.vocab)}")
    
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
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.char2idx.get('<PAD>', 0)).to(self.device)
        self.reset_state()
    
    def reset_state(self):
        self.hidden_state = None
    
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
            # 调整目标序列长度以匹配输入序列
            target_seq = target_seq[:input_length]
            target_length = len(target_seq)
        
        # 转换为张量
        inputs = torch.tensor([full_input], dtype=torch.long).to(self.device)
        targets = torch.tensor([target_seq], dtype=torch.long).to(self.device)
        
        # 训练循环
        self.model.train()
        last_loss = 0.0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            
            # 确保输出和目标形状匹配
            if outputs.shape[1] != targets.shape[1]:
                # 截取输出以匹配目标长度
                outputs = outputs[:, :targets.shape[1], :]
            
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
            last_loss = loss.item()
            
            loss.backward()
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
            
            # 保持在 CPU 上（关键修正：不在预处理时移到 GPU）
            input_seqs.append(torch.tensor(full_input, dtype=torch.long))
            target_seqs.append(torch.tensor(target_seq, dtype=torch.long))
        
        # 2. 批量填充（CPU 上操作）
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_seqs, batch_first=True, padding_value=pad_idx
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=pad_idx
        )
        
        # 3. DataLoader 处理 CPU 张量（pin_memory 有效）
        dataset = torch.utils.data.TensorDataset(padded_inputs, padded_targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True  # 此时张量在 CPU 上，pin_memory 有效
        )
        
        # 训练循环
        self.model.train()
        total_loss = 0.0
        device = self.device  # 保存设备信息
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                # 关键修正：在 batch 加载时移到 GPU（利用 pin_memory 加速传输）
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs, _ = self.model(batch_inputs)
                
                # 计算损失（利用 ignore_index 忽略 PAD）
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    batch_targets.reshape(-1)
                )
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
        
        return total_loss / epochs

    def sliding_window_train(self, text, study_lr=0.001, epochs=1, batch_size=32, step_ratio=0.5):
        encoded_text = self._encode_text(text)
        total_length = len(encoded_text)
        samples = []
        step = max(int(self.max_length * step_ratio), 1)  # 步长至少为1
        if total_length > 100000:  # 超长文本使用更大的步长
            step = max(int(self.max_length * step_ratio * 2), 1)
        min_valid_length = 2  # 窗口至少需要2个字符才能生成有效样本（prompt和target各至少1个）
        
        if total_length < min_valid_length:
            print(f"文本过短（长度{total_length}），无法生成有效样本")
            return 0.0  # 直接返回，避免训练
        
        # 短文本处理（长度≤max_length但≥2）
        if total_length <= self.max_length:
            prompt = encoded_text[:-1]  # 前n-1个字符
            target = encoded_text[1:]   # 后n-1个字符
            # 确保prompt和target非空
            if len(prompt) > 0 and len(target) > 0:
                prompt_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in prompt)
                target_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in target)
                samples.append((prompt_text, target_text))
            print(f"生成1个有效样本（长度{total_length}）")
        else:
            # 长文本滑动窗口
            start = 0
            while start + min_valid_length <= total_length:  # 确保窗口至少能生成有效样本
                end = min(start + self.max_length, total_length)
                window = encoded_text[start:end]
                
                # 窗口长度至少为2，才能生成非空prompt和target
                if len(window) < min_valid_length:
                    start += step
                    continue
                
                prompt = window[:-1]
                target = window[1:]
                
                # 再次检查prompt和target非空（双重保险）
                if len(prompt) == 0 or len(target) == 0:
                    start += step
                    continue
                
                prompt_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in prompt)
                target_text = ''.join(self.idx2char.get(idx, '<UNK>') for idx in target)
                samples.append((prompt_text, target_text))
                
                # 移动窗口，避免重复
                start += step
                
                # 最后一个窗口覆盖结尾
                if end == total_length:
                    break
        
        # 过滤空样本（终极保险）
        valid_samples = []
        for p, t in samples:
            if len(p) > 0 and len(t) > 0:
                valid_samples.append((p, t))
        
        if not valid_samples:
            print("未生成有效样本，跳过训练")
            return 0.0
        
        print(f"生成 {len(valid_samples)} 个有效训练样本（过滤掉{len(samples)-len(valid_samples)}个无效样本）")
        return self.train_batch(valid_samples, study_lr, epochs, batch_size)
    
    def generate(self, prompt, temperature=0.7, length=50):
        if self.model is None:
            raise ValueError("模型未初始化")
            
        self.model.eval()
        input_seq = self._encode_text(prompt)
        generated = []
        
        with torch.no_grad():
            for i in range(length):
                inputs = torch.tensor([input_seq], dtype=torch.long).to(self.device)
                outputs, self.hidden_state = self.model(inputs, self.hidden_state)
                last_output = outputs[0, -1, :]
                
                if temperature > 0:
                    probs = torch.softmax(last_output / temperature, dim=-1)
                    next_char_idx = torch.multinomial(probs, 1).item()
                else:
                    next_char_idx = torch.argmax(last_output).item()
                
                generated.append(next_char_idx)
                input_seq.append(next_char_idx)
                
                if len(input_seq) > self.max_length:
                    input_seq = input_seq[-self.max_length:]
                
                char = self.idx2char.get(next_char_idx, '<UNK>')
                if char == '<S>' or char == '<EOS>':
                    break
        
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
            'max_length': self.max_length
        }, save_path)
        print(f"[模型] 已保存到 {save_path}")
    
    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件 {load_path} 不存在")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.char2idx = checkpoint['char2idx']
        self.idx2char = checkpoint['idx2char']
        self.embedding_dim = checkpoint.get('embedding_dim', 512)
        self.hidden_dim = checkpoint.get('hidden_dim', 2048)
        self.max_length = checkpoint.get('max_length', 128)
        
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.reset_state()
        print(f"[模型] 已从 {load_path} 加载")

class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden_state=None):
        emb = self.embedding(x)
        if hidden_state is not None:
            lstm_out, hidden_state = self.lstm(emb, hidden_state)
        else:
            lstm_out, hidden_state = self.lstm(emb)
        output = self.fc(lstm_out)
        return output, hidden_state