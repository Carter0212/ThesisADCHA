#!/bin/bash

# 定義要執行的程式清單
FWAs=(10 20 40 50)
Algs=("ADCHA" "HHO" "HHO_SMA" "NSGA" "Random")
#Algs=("NSGA")
Thrs=(4 6 8 10 12)
Iteration=5000
# 遍歷每個程式
for FWA in "${FWAs[@]}"; do
    for Alg in "${Algs[@]}"; do
        echo "執行程式：Choice_Algs.py -BS 5 -FWA $FWA -Thr 6 -unit Mbps"
        # 執行程式並獲取輸出
        python3 Choice_Algs.py -BS 5 -FWA "$FWA" -Thr 6 -unit Mbps -iter "$Iteration" -type "$Alg"
    done
done
for Alg in "${Algs[@]}"; do
    for Thr in "${Thrs[@]}"; do
        echo "執行程式：Times.py -BS 5 -FWA $FWA -Thr $Thr -unit Mbps" -iter "$Iteration" -type "$Alg"
        # 執行程式並獲取輸出
        python3 Choice_Algs.py -BS 5 -FWA 30 -Thr "$Thr" -unit Mbps -iter "$Iteration" -type "$Alg"

    done
done

    # python3 Choice_Algs.py -type $script -BS $BS -FWA $fwa -Thr $Thr  -start $startTime -end $endTime -unit $unit  -iter $iteration"
