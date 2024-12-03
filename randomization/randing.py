import torch
import numpy as np
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

# 讀取原始權重
model_data = torch.load(script_dir+'/c3d-pretrained.pth', weights_only=True)


# 開啟txt檔案來寫入
with open('model_random.txt', 'w', encoding='utf-8') as f:
    
    # 處理所有權重層
    for name, param in model_data.items():
        if 'weight' in name:
            # 獲取原始形狀
            original_shape = param.shape
            
            # 生成 -128 到 127 之間的隨機整數
            random_weights = np.random.randint(-128, 128, size=original_shape)
            
            # 轉換回tensor並更新
            model_data[name] = torch.FloatTensor(random_weights)
            
            # 寫入txt
            f.write(f"\nlayer: {name}\n")
            f.write(f"shape: {original_shape}\n")
            f.write("weights: \n")
            f.write(str(random_weights))
            f.write("\n" + "-"*50 + "\n")
            
        elif 'bias' in name:
            # 寫入bias資訊
            f.write(f"\nlayer: {name}\n")
            f.write(f"shape: {param.shape}\n")
            f.write("Bias: \n")
            f.write(str(param.numpy()))
            f.write("\n" + "-"*50 + "\n")

# 同時也保存為pth檔
torch.save(model_data, 'model_random.pth')

print("Done.")