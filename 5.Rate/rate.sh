
python_scripts=("ADCHA" "HHO" "HHO_SMA" "NSGA" "Random")
# Scene=BS5_FWA50_1Gbps
BS=5
fwa=30
Thr=(4 6 8 10 12)
#Thr=(4)
unit="Mbps"
iteration=5000
startTime=0
endTime=500


for thr in "${Thr[@]}"; do

    if [ "$thr" == "2.5" ]; then
        Scenes="S${startTime}E${endTime}_${fwa}_2_5"
    else
        Scenes="S${startTime}E${endTime}_${fwa}_${thr}"
    fi
    


    # if [ "$thr" == "6" ]; then
    #     unit="Mbps"
    # else
    #     unit="Gbps"
    # fi

    tmux has-session -t $Scenes 2>/dev/null
    if [ $? == 0 ]; then
        tmux kill-session -t $Scenes
    fi

    tmux new-session -d -s $Scenes

    for script in "${python_scripts[@]}"; do
        # echo "Creating window: $script in session: $Scenes"
        tmux new-window -t $Scenes -n "$script"

        # echo "Sending keys to window: $script in session: $Scenes"
        tmux send-keys -t $Scenes:"$script" "tmux rename-window '$script'; python3 Choice_Algs.py -type $script -BS $BS -FWA $fwa -Thr $thr -unit $unit -start $startTime -end $endTime -iter $iteration" C-m
        sleep 0.5
    done

    gnome-terminal -- tmux attach -t $Scenes &
done

