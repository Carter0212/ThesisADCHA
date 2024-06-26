# 指定需要运行的 Python 3 脚本列表
python_scripts=("ADCHA" "HHO" "HHO_SMA" "NSGA" "Random")
# python_scripts=("HHO_DE" "HHO_Chaos" "Chaos" "DE")
# python_scripts=("ADCHA")
BS=5
# FWA=(10 20 30 40 50)
FWA=(20)
Thr=6
unit="Mbps"
iteration=50000
startTime=0
endTime=1

# 遍历 FWA 数组
for fwa in "${FWA[@]}"; do
    # 创建一个新的 tmux 会话名称
    Scenes="S${startTime}E${endTime}_${fwa}_${Thr}"

    # 如果会话已存在，杀死会话并创建新会话
    tmux has-session -t $Scenes 2>/dev/null
    if [ $? == 0 ]; then
        tmux kill-session -t $Scenes
    fi

    # 创建一个新会话
    tmux new-session -d -s $Scenes

    # 遍历脚本列表并在不同的窗口中运行它们
    for script in "${python_scripts[@]}"; do
        # 创建一个新窗口并设置窗格名称
        tmux new-window -t $Scenes -n "${script}_S${startTime}E${endTime}"

        # 在新窗格中设置窗格名称并运行 Python 3 脚本
        tmux send-keys -t $Scenes:"$script" "tmux rename-window '$script'; python3 Choice_Algs.py -type $script -BS $BS -FWA $fwa -Thr $Thr  -start $startTime -end $endTime -unit $unit  -iter $iteration" C-m
        
        sleep 0.5
    done

    # 在新的终端窗口中打开 tmux 会话
    gnome-terminal -- tmux attach -t $Scenes &
done
