import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

current_date = '2024-04-28'
FWAs = [10, 20, 30, 40, 50]
Algs = ['ADCHA', 'HHO', 'HHO_SMA', 'NSGAII','Random']
file_name = 'ExecutionTime.csv'

# Initialize a dictionary to store all data
all_data = {}

for FWA in FWAs:
    scenes = f'5BS_{FWA}FWA_GoalThr6.0Mbps'
    all_data[scenes] = {}
    for Alg in Algs:
        Init_Path = f'./output/{scenes}/{current_date}/{Alg}'
        execution_time_file = os.path.join(Init_Path, file_name)
        
        # Read the CSV file
        scene_data = pd.read_csv(execution_time_file, header=None)
        
        # Store the mean execution time for this algorithm
        all_data[scenes][Alg] = scene_data.mean().values[0]

# Convert the dictionary to a DataFrame
df = pd.DataFrame(all_data)

# Process column names to keep only important FWA information
df.columns = [int(col.split('_')[1][:2]) for col in df.columns]

# Plot the data as a bar chart
plt.figure(figsize=(8, 4))  # Set the figure size
ax = df.plot(kind='bar')

# Set the title and labels
# ax.set_title('Execution Time for Different Number of FWAs and Algorithms', fontsize=12)
ax.set_xlabel('Algorithms', fontsize=15)
ax.set_ylabel('Execution Time (s)', fontsize=15)

# Set font for tick labels
ax.tick_params(axis='both', which='major', labelsize=8)

# Adjust x-axis labels to be horizontal
plt.xticks(rotation=0)

# Adjust legend
handles, labels = ax.get_legend_handles_labels()
labels = [f'{label}FWA' for label in labels]
ax.legend(handles, labels, fontsize=8)

# Show the plot
plt.tight_layout()
plt.savefig(f'./FWA_Execution.jpg')

# Initialize a dictionary to store all data
all_data = {}
Thrs = [4.0, 6.0 ,8.0, 10.0]
for thr in Thrs:
    scenes = f'5BS_30FWA_GoalThr{thr}Mbps'
    # if :
    #     scenes = f'5BS_30FWA_GoalThr{thr}Mbps'
    # else:
    #     scenes = f'5BS_30FWA_GoalThr{thr}Gbps'
    all_data[scenes] = {}
    for Alg in Algs:
        Init_Path = f'./output/{scenes}/{current_date}/{Alg}'
        execution_time_file = os.path.join(Init_Path, file_name)
        
        # Read the CSV file
        scene_data = pd.read_csv(execution_time_file, header=None)
        
        # Store the mean execution time for this algorithm
        all_data[scenes][Alg] = scene_data.mean().values[0]

# Convert the dictionary to a DataFrame
df = pd.DataFrame(all_data)

# Process column names to keep only important FWA information
df.columns = [int(col.split('_')[1][:2]) for col in df.columns]

# Plot the data as a bar chart
plt.figure(figsize=(8, 4))  # Set the figure size
# ax = df.plot(kind='bar')

# # Set the title and labels
# ax.set_title('Execution Time for Different data rate requirments and Algorithms', fontsize=12)
# ax.set_xlabel('Algorithms', fontsize=15)
# ax.set_ylabel('Execution Time(s)', fontsize=15)

# # Set font for tick labels
# ax.tick_params(axis='both', which='major', labelsize=8)

# # Adjust legend
# handles, labels = ax.get_legend_handles_labels()
# # labels = ['600Mbps' if label == 6.0 else f'{int(label)}Mbps' for label in Thrs]
# # labels = ['400Mbps' if label == 4.0 else f'{int(label)}Mbps' for label in Thrs]
# # labels = ['800Mbps' if label == 8.0 else f'{int(label)}Mbps' for label in Thrs]
# # labels = ['1Gbps' if label == 10.0 else f'{int(label)}Gbps' for label in Thrs]
# labels = ['600Mbps' if label == 6.0 else '400Mbps' if label == 4.0 else '800Mbps' if label == 8.0 else '1Gbps' if label == 10.0 else f'{int(label)}Mbps' if label < 10 else f'{int(label)}Gbps' for label in Thrs]

# # labels = ['600Mbps' if label == 6.0 else f'{int(label)}Gbps' for label in Thrs]
# ax.legend(handles, labels, fontsize=8)

# # Show the plot
# plt.tight_layout()
# plt.savefig(f'./thr_Execution.jpg')


ax = df.plot(kind='bar')

# Set the title and labels
# ax.set_title('Execution Time for Different data rate requirments and Algorithms', fontsize=12)
ax.set_xlabel('Algorithms', fontsize=15)
ax.set_ylabel('Execution Time (s)', fontsize=15)

# Set font for tick labels
ax.tick_params(axis='both', which='major', labelsize=8)

# Adjust x-axis labels to be horizontal
plt.xticks(rotation=0)

# Adjust legend
handles, labels = ax.get_legend_handles_labels()
labels = ['600Mbps' if label == 6.0 else '400Mbps' if label == 4.0 else '800Mbps' if label == 8.0 else '1Gbps' if label == 10.0 else f'{int(label)}Mbps' if label < 10 else f'{int(label)}Gbps' for label in Thrs]
ax.legend(handles, labels, fontsize=8)

# Show the plot
plt.tight_layout()
plt.savefig(f'./thr_Execution.jpg')