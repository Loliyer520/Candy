import json
import os
from candyfish import Model

def print_menu():
    """打印操作菜单"""
    print("\n===== 对话模型操作菜单 =====")
    print("l - 加载模型")
    print("s - 保存模型")
    print("t - 训练模型")
    print("c - 设置学习率和训练次数")
    print("g - 测试对话生成")
    print("a - 人工添加对话并训练")
    print("q - 退出程序")
    print("==========================")

def load_training_samples(trains_dir='train/dialogs', max_length=256):
    """加载训练样本，自动截断超过max_length的上下文"""
    samples = []
    
    if not os.path.exists(trains_dir):
        print(f"错误: 训练文件夹 {trains_dir} 不存在")
        return []
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(trains_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"警告: 在 {trains_dir} 中未找到任何JSON文件")
        return []
    
    # 列出所有找到的JSON文件
    print("\n找到以下训练文件:")
    for i, filename in enumerate(json_files, 1):
        print(f"{i}. {filename}")
    
    # 构建训练样本
    for filename in json_files:
        filepath = os.path.join(trains_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            # 检查是否为列表格式的对话数据
            if not isinstance(conversation, list):
                print(f"警告: {filename} 不是对话列表格式，跳过")
                continue
            
            # 使用列表存储对话历史，便于截断
            context_parts = []
            
            # 逐轮处理对话
            for i in range(0, len(conversation), 2):
                # 确保有用户和AI的两轮对话
                if i+1 >= len(conversation):
                    break
                
                user_msg = conversation[i]
                ai_msg = conversation[i+1]
                
                # 验证角色
                if user_msg.get('role') != 'user' or ai_msg.get('role') != 'assistant':
                    print(f"警告: {filename} 第{i}轮对话角色错误，跳过")
                    continue
                
                user_content = user_msg.get('content', '').strip()
                ai_content = ai_msg.get('content', '').strip()
                
                if not user_content or not ai_content:
                    continue
                
                # 添加用户消息到上下文
                user_part = f"用户：{user_content}<S>"
                ai_part = f"AI：{ai_content}<S>"
                
                # 临时构建当前上下文
                temp_context = ''.join(context_parts) + user_part + "AI："
                temp_full_length = len(temp_context) + len(ai_content) + len("<S>")
                
                # 如果超过最大长度，移除最早的对话轮次
                while temp_full_length > max_length and len(context_parts) > 0:
                    # 移除最早的对话轮次（一对用户+AI）
                    if len(context_parts) >= 2:
                        # 移除最早的一对对话
                        context_parts.pop(0)  # 用户部分
                        context_parts.pop(0)  # AI部分
                    else:
                        # 如果只有一部分，也移除
                        context_parts.pop(0)
                    
                    # 重新计算长度
                    temp_context = ''.join(context_parts) + user_part + "AI："
                    temp_full_length = len(temp_context) + len(ai_content) + len("<S>")
                
                # 检查是否仍然超长（可能当前单轮对话就超长）
                if temp_full_length > max_length:
                    print(f"警告: 单轮对话长度({temp_full_length})超过max_length({max_length})，跳过该轮对话")
                    continue
                
                # 创建训练样本
                prompt = ''.join(context_parts) + user_part + "AI："
                target = f"{ai_content}<S>"
                samples.append((prompt, target))
                
                # 将本轮对话添加到上下文，用于后续对话
                context_parts.append(user_part)
                context_parts.append(ai_part)
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    return samples

def train_dialogue_model():
    # 初始化模型和参数
    model = Model()
    is_model_initialized = False
    learning_rate = 0.001  # 默认学习率
    total_epochs = 50      # 默认训练轮次
    max_length = 256       # 默认最大长度
    
    while True:
        print_menu()
        choice = input("请选择操作 (l/s/t/c/g/a/q): ").strip().lower()
        
        if choice == 'l':
            # 加载模型
            load_model_name = input("请输入要加载的模型名称: ").strip()
            if load_model_name:
                model_path = f'data/{load_model_name}.spt'
                try:
                    model.load(model_path)
                    is_model_initialized = True
                    print(f"成功加载模型: {model_path}")
                except Exception as e:
                    print(f"加载模型失败: {e}")
            else:
                print("模型名称不能为空")
        
        elif choice == 's':
            # 保存模型
            if not is_model_initialized:
                print("请先初始化或加载模型")
                continue
                
            while True:
                save_name = input("请输入保存模型的名称: ").strip()
                if save_name:
                    save_path = f'data/{save_name}.spt'
                    try:
                        # 确保data文件夹存在
                        os.makedirs('data', exist_ok=True)
                        model.save(save_path)
                        print(f"模型已保存至: {save_path}")
                        break
                    except Exception as e:
                        print(f"保存模型失败: {e}，请重新输入名称")
                else:
                    print("名称不能为空，请重新输入")
        
        elif choice == 't':
            # 训练模型
            if not is_model_initialized:
                # 初始化新模型
                print("未检测到模型，正在初始化新模型...")
                model.new(embedding_dim=256, hidden_dim=2048, max_length=max_length)
                is_model_initialized = True
            
            # 加载训练样本
            samples = load_training_samples(max_length=max_length)
            
            if not samples:
                print("没有可用的训练样本，无法进行训练")
                continue
            
            print(f"共加载 {len(samples)} 个训练样本")
            
            # 询问是否继续
            confirm = input("\n是否开始训练? (y/n): ").strip().lower()
            if confirm != 'y':
                print("已取消训练")
                continue
            
            # 训练循环
            for epoch in range(1):
                epoch_loss = 0.0
                sample_count = 0
                
                for prompt, target in samples:
                    loss = model.train(
                        prompt=prompt,
                        target=target,
                        study_lr=learning_rate,
                        epochs=total_epochs  # 每个样本训练一次
                    )
                    epoch_loss += loss
                    sample_count += 1
                
                avg_loss = epoch_loss / sample_count
                print(f"Epoch {epoch+1}/{total_epochs} - 平均损失: {avg_loss:.4f}")
            
            print("\n所有样本训练完成")
        
        elif choice == 'c':
            # 设置学习率、训练次数和最大长度
            try:
                lr_input = input(f"请输入学习率 (当前: {learning_rate}): ").strip()
                if lr_input:
                    learning_rate = float(lr_input)
                    if learning_rate <= 0:
                        raise ValueError("学习率必须大于0")
                
                epochs_input = input(f"请输入训练轮次 (当前: {total_epochs}): ").strip()
                if epochs_input:
                    total_epochs = int(epochs_input)
                    if total_epochs <= 0:
                        raise ValueError("训练轮次必须大于0")
                
                # 新增最大长度设置
                max_input = input(f"请输入最大长度 (当前: {max_length}): ").strip()
                if max_input:
                    max_length = int(max_input)
                    if max_length <= 0:
                        raise ValueError("最大长度必须大于0")
                
                print(f"设置成功 - 学习率: {learning_rate}, 训练轮次: {total_epochs}, 最大长度: {max_length}")
            except ValueError as e:
                print(f"输入错误: {e}，保持原有设置")
        
        elif choice == 'g':
            # 测试对话生成
            if not is_model_initialized:
                print("请先初始化或加载模型")
                continue
            
            user_input = input("请输入用户消息: ").strip()
            if not user_input:
                print("用户消息不能为空")
                continue
            
            # 构建提示
            prompt = f"用户：{user_input}<S>\nAI："
            
            # 温度参数，默认0.5
            while True:
                temp_input = input(f"请输入温度 (0.0-1.0，默认: 0.5): ").strip()
                if not temp_input:  # 使用默认值
                    temperature = 0.5
                    break
                try:
                    temperature = float(temp_input)
                    if 0.0 <= temperature <= 1.0:
                        break
                    else:
                        print("温度必须在0.0到1.0之间，请重新输入")
                except ValueError:
                    print("请输入有效的数字")
            
            # 生成长度，默认100
            while True:
                len_input = input(f"请输入回复长度 (默认: 100): ").strip()
                if not len_input:  # 使用默认值
                    length = 100
                    break
                try:
                    length = int(len_input)
                    if length > 0:
                        break
                    else:
                        print("长度必须大于0，请重新输入")
                except ValueError:
                    print("请输入有效的整数")
            
            # 生成回复
            print("\n生成中...")
            generated = model.generate(prompt=prompt, temperature=temperature, length=length)
            print(f"\nAI回复: \n{prompt}\033[34m{generated}\033[0m\n")

            input("按任意键继续...")
        
        elif choice == 'a':
            # 人工添加对话并训练
            if not is_model_initialized:
                # 初始化新模型
                print("未检测到模型，正在初始化新模型...")
                model.new(embedding_dim=256, hidden_dim=2048, max_length=max_length)
                is_model_initialized = True
            
            # 获取人工输入的对话
            user_msg = input("请输入用户发言: ").strip()
            if not user_msg:
                print("用户发言不能为空")
                continue
                
            ai_msg = input("请输入AI回复: ").strip()
            if not ai_msg:
                print("AI回复不能为空")
                continue
            
            # 构建训练样本
            prompt = f"用户：{user_msg}<S>AI："
            target = f"{ai_msg}<S>"
            
            # 获取学习率
            while True:
                lr_input = input(f"请输入学习率 (默认: {learning_rate}): ").strip()
                if not lr_input:  # 使用默认值
                    lr = learning_rate
                    break
                try:
                    lr = float(lr_input)
                    if lr > 0:
                        break
                    else:
                        print("学习率必须大于0，请重新输入")
                except ValueError:
                    print("请输入有效的数字")
            
            # 获取训练次数，默认20
            while True:
                epochs_input = input("请输入训练次数 (默认: 20): ").strip()
                if not epochs_input:  # 使用默认值
                    epochs = 20
                    break
                try:
                    epochs = int(epochs_input)
                    if epochs > 0:
                        break
                    else:
                        print("训练次数必须大于0，请重新输入")
                except ValueError:
                    print("请输入有效的整数")
            
            # 进行训练
            print(f"\n开始训练，共{epochs}次...")
            loss = model.train(prompt=prompt, target=target, study_lr=lr, epochs=epochs)

            print(f"训练完成! Loss: {loss:.4f}")
        
        elif choice == 'q':
            print("程序已退出")
            break
        
        else:
            print("无效的选择，请重试")

if __name__ == "__main__":
    train_dialogue_model()