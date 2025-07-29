import os
from candyfish import Model
import chardet

EMBEDDING_DIM = 256
HIDDEN_DIM = 2048
MAX_LENGTH = 1024

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节用于检测
    result = chardet.detect(raw_data)
    return result['encoding']

def read_text_file(file_path):
    """读取文本文件，自动处理编码问题"""
    encoding = detect_encoding(file_path)
    # 处理可能的编码检测错误
    if encoding is None:
        encoding = 'utf-8'
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # 尝试常见的其他编码
        for enc in ['gbk', 'gb2312', 'utf-16', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # 如果所有尝试都失败，返回空字符串
        print(f"警告：无法正确读取文件 {file_path} 的内容")
        return ""

def print_menu():
    """打印菜单"""
    print("\n===== 模型操作菜单 =====")
    print("l - 加载模型")
    print("s - 保存模型")
    print("t - 训练模型")
    print("c - 设置学习率和训练次数")
    print("g - 测试生成文本")
    print("q - 退出程序")
    print("========================")

def main():
    # 初始化模型和参数
    model = Model()
    is_model_initialized = False
    study_lr = 0.001  # 默认学习率
    epochs = 5        # 默认训练轮次
    
    while True:
        print_menu()
        choice = input("请选择操作 (l/s/t/c/g/q): ").strip().lower()
        
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
                model.new(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, max_length=MAX_LENGTH)
                is_model_initialized = True
            
            # 检查train/text文件夹是否存在
            text_dir = 'train/text'
            if not os.path.exists(text_dir):
                print(f"错误：文件夹 {text_dir} 不存在")
                continue
            
            # 获取所有txt文件
            txt_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
            if not txt_files:
                print(f"警告：在 {text_dir} 文件夹中未找到任何txt文件")
                continue
            
            # 列出所有txt文件
            print("\n找到以下训练文件:")
            for i, file_name in enumerate(txt_files, 1):
                print(f"{i}. {file_name}")
            
            # 询问是否继续
            confirm = input("\n是否开始训练? (y/n): ").strip().lower()
            if confirm != 'y':
                print("已取消训练")
                continue
            
            # 依次训练每个txt文件
            total_files = len(txt_files)
            for i, file_name in enumerate(txt_files, 1):
                file_path = os.path.join(text_dir, file_name)
                print(f"\n正在处理文件 {i}/{total_files}: {file_name}")
                
                # 读取文件内容
                text = read_text_file(file_path)
                if not text:
                    print(f"跳过文件 {file_name}，内容为空或无法读取")
                    continue
                
                # 训练模型
                print(f"开始训练 {file_name} ...")
                loss = model.sliding_window_train(text=text, study_lr=study_lr, epochs=epochs)
                print(f"{file_name} 训练完成，损失: {loss:.4f}")
            
            print("\n所有文件训练完成")
        
        elif choice == 'c':
            # 设置学习率和训练次数
            try:
                lr_input = input(f"请输入学习率 (当前: {study_lr}): ").strip()
                if lr_input:
                    study_lr = float(lr_input)
                    if study_lr <= 0:
                        raise ValueError("学习率必须大于0")
                
                epochs_input = input(f"请输入训练轮次 (当前: {epochs}): ").strip()
                if epochs_input:
                    epochs = int(epochs_input)
                    if epochs <= 0:
                        raise ValueError("训练轮次必须大于0")
                
                print(f"设置成功 - 学习率: {study_lr}, 训练轮次: {epochs}")
            except ValueError as e:
                print(f"输入错误: {e}，保持原有设置")
        
        elif choice == 'g':
            # 测试生成文本
            if not is_model_initialized:
                print("请先初始化或加载模型")
                continue
            
            prompt = input("请输入生成的起始文本: ").strip()
            if not prompt:
                print("起始文本不能为空")
                continue
            
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
                len_input = input(f"请输入续写长度 (默认: 100): ").strip()
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
            
            # 生成文本
            print("\n生成中...")
            generated = model.generate(prompt=prompt, temperature=temperature, length=length)
            print(f"\n生成结果: \033[34m{prompt}{generated}\033[0m\n")

            input("按任意键继续...")
        
        elif choice == 'q':
            print("程序已退出")
            break
        
        else:
            print("无效的选择，请重试")

if __name__ == "__main__":
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    MAX_LENGTH = 512
    main()
