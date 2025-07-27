from candyfish import Model

# 初始化模型
model = Model()
model.new(embedding_dim=512, hidden_dim=2048, max_length=1024)
# model.load('data/test.spt')

# 单次训练
print(f"训练损失: {model.train(prompt='你好 Ai:', target='你好呀！', study_lr=0.001, epochs=50):.4f}")

# 测试生成
print(f"生成续写: 你好{model.generate(prompt='你好', temperature=0.1, length=100)}")

# 批训练模型
print(f"训练损失: {model.train_batch(
        samples=[
            ('你好 Ai:', '你好呀！很高兴能和你交流~'),
            ('你是谁 Ai:', '我是豆包，是字节跳动公司开发的人工智能助手。')], 
            study_lr=0.001, epochs=50):.4f}")

# 生成文本
print(f"生成续写: {model.generate(prompt='你好', temperature=0.7, length=100)}")

# 续写训练
print(f"训练损失: {model.sliding_window_train(text='生成式人工智能是笨蛋', study_lr=0.001, epochs=50):.4f}")

# 生成文本
print(f"生成续写: {model.generate(prompt='生成式', temperature=0.1, length=100)}")

# 保存和加载模型
model.save('data/test.spt')