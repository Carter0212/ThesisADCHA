# 指定需要运行的 Python 3 脚本列表
python_scripts=("ADCHA" "HHO" "HHO_SMA" "NSGA" "Random")
# Scene=BS5_FWA50_1Gbps
BS=5
fwa=30
#Thr=(4 6 8 10 12)
Thr=(4)
unit="Mbps"
iteration=5000
startTime=0
endTime=1

# 遍历 FWA 数组
# 遍历 Thr 数组
for thr in "${Thr[@]}"; do
    # 创建一个新的 tmux 会话名称
    # Scenes="S${startTime}E${endTime}_${fwa}_${Thr}"
    if [ "$thr" == "2.5" ]; then
        Scenes="S${startTime}E${endTime}_${fwa}_2_5"
    else
        Scenes="S${startTime}E${endTime}_${fwa}_${thr}"
    fi
    

    # 如果 Thr 为 0.6，将 unit 设置为 "Mbps"
    # if [ "$thr" == "6" ]; then
    #     unit="Mbps"
    # else
    #     unit="Gbps"
    # fi

    # 如果会话已存在，杀死会话并创建新会话
    tmux has-session -t $Scenes 2>/dev/null
    if [ $? == 0 ]; then
        tmux kill-session -t $Scenes
    fi

    # # 创建一个新会话
    tmux new-session -d -s $Scenes

    # 遍历脚本列表并在不同的窗口中运行它们
    for script in "${python_scripts[@]}"; do
        # echo "Creating window: $script in session: $Scenes"
        tmux new-window -t $Scenes -n "$script"

        # 在新窗格中设置窗格名称并运行 Python 3 脚本
        # echo "Sending keys to window: $script in session: $Scenes"
        tmux send-keys -t $Scenes:"$script" "tmux rename-window '$script'; python3 Choice_Algs.py -type $script -BS $BS -FWA $fwa -Thr $thr -unit $unit -start $startTime -end $endTime -iter $iteration" C-m
        sleep 0.5
    done

    # 在新的终端窗口中打开 tmux 会话
    gnome-terminal -- tmux attach -t $Scenes &
done

