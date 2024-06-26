
python_scripts=("ADCHA" "HHO" "HHO_SMA" "NSGA" "Random")
# python_scripts=("HHO_DE" "HHO_Chaos" "Chaos" "DE")
# python_scripts=("ADCHA")
BS=5
# FWA=(10 20 30 40 50)
FWA=(10 20 30 40 50)
Thr=6
unit="Mbps"
iteration=5000
startTime=0
endTime=500

for fwa in "${FWA[@]}"; do

    Scenes="S${startTime}E${endTime}_${fwa}_${Thr}"

    tmux has-session -t $Scenes 2>/dev/null
    if [ $? == 0 ]; then
        tmux kill-session -t $Scenes
    fi

    tmux new-session -d -s $Scenes


    for script in "${python_scripts[@]}"; do

        tmux new-window -t $Scenes -n "${script}_S${startTime}E${endTime}"

        tmux send-keys -t $Scenes:"$script" "tmux rename-window '$script'; python3 Choice_Algs.py -type $script -BS $BS -FWA $fwa -Thr $Thr  -start $startTime -end $endTime -unit $unit  -iter $iteration" C-m
        
        sleep 0.5
    done

    gnome-terminal -- tmux attach -t $Scenes &
done
