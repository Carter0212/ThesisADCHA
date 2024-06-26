import pickle
import matplotlib.pyplot as plt
import numpy as np
# 读取pickle文件
with open('5BS_40FWA_coords.pkl', 'rb') as file:
    df = pickle.load(file)
FWA_coords = np.array(df['FWA_coords'][0])
BS_coords = np.array(df['BS_coords']).squeeze()
print(df['BS_coords'])
FWA_x = FWA_coords[:, 0]
FWA_y = FWA_coords[:, 1]
# print(BS_coords[0])
BS_x = BS_coords[:, 0]
BS_y = BS_coords[:, 1]

# 绘制FWA和BS的座标
plt.scatter(FWA_x, FWA_y, color='blue', label='FWA')
plt.scatter(BS_x, BS_y, color='red', marker='s', label='SBS')


plt.xlim(0, 100)
plt.ylim(0, 100)
# 添加图例和标题
plt.grid(True)

plt.legend()

plt.savefig('topo.png')

# 显示图形
plt.show()