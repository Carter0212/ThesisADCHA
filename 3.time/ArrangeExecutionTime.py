import os
import pandas as pd

# 指定包含所有演算法的父目錄
parent_dir = "./2024-04-28"

# 初始化一個空的 DataFrame 來保存所有數據
all_data = pd.DataFrame()

# 遍歷所有演算法的目錄
for algorithm_dir in os.listdir(parent_dir):
    algorithm_path = os.path.join(parent_dir, algorithm_dir)
    
    # 確保這是一個目錄
    if os.path.isdir(algorithm_path):
        
        # 初始化一個空的 DataFrame 來保存當前演算法的所有數據
        algorithm_data = pd.DataFrame()
        
        # 遍歷當前演算法目錄下的所有場景目錄
        for scene_dir in os.listdir(algorithm_path):
            scene_path = os.path.join(algorithm_path, scene_dir)
            
            # 確保這是一個目錄
            if os.path.isdir(scene_path):
                
                # 構建 ExecutionTime.csv 文件的路徑
                execution_time_file = os.path.join(scene_path, "ExecutionTime.csv")
                
                # 確保文件存在
                if os.path.isfile(execution_time_file):
                    
                    # 讀取 ExecutionTime.csv 文件並將其加入到當前演算法的數據中
                    scene_data = pd.read_csv(execution_time_file, header=None, names=[scene_dir])
                    algorithm_data = pd.concat([algorithm_data, scene_data], axis=1)
        
        # 添加演算法名稱作為索引的備註
        algorithm_data.index = pd.MultiIndex.from_product([[algorithm_dir], algorithm_data.index], names=['Algorithm', 'Scene'])
        
        # 將當前演算法的數據加入到所有數據中
        all_data = pd.concat([all_data, algorithm_data], axis=0)

# 將整理好的數據保存到一個新的 CSV 文件中
all_data.to_csv("all_execution_times.csv")

FWAs=[10,20,30,40,50]
Thrs=['600Mbps','1Gbps','2Gbps','2.5Gbps','3Gbps']
Algs=['ADCHA','HHO','HHO_SMA','NSGAII']

for FWA in FWAs:
    scenes = f'5BS_{FWA}FWA_GoalThr1.0Gbps'
    
