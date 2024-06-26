Follow this hackmd note
https://hackmd.io/3DQQig-YR9SWxFSACmJ4eg?both#%E5%9F%B7%E8%A1%8C%E7%A8%8B%E5%BC%8F


# ADCHA Readme


## A. Environmental Setting
| Ubuntu Version | Python3 Version |
| -------------- | --------------- |
| 22.04           | 3.9 & 3.10            |

目前程式已證實可以在以上環境版本皆可執行

**Setup Python Module**

安裝Module

``` command
sudo apt-get update
sudo apt-get install python3-pip
```


安裝Module

``` command
pip3 install -r requirements.txt
```

安裝tmux

``` command
sudo apt-get install tmux
```
[tmux使用教學](https://hackmd.io/@Cheng-Hao/Hyk9f6mZd)
## B. File introduction

以下檔案**建議使用tmux來測試**,因為會花費大量時間,以免因為當機或誤觸導致執行中斷。

1. test：測試電腦是否能夠運行本篇演算法
2. deploy：模擬場景SBS和FWA的位置
3. FWA：模擬固定FWA不同傳輸速率需求的結果
4. rate：模擬固定傳輸速率需求不同FWA的結果
5. time：模擬所花費的執行時間
6. appendix：執行附錄中所有的算法


### 1. test
該步驟是為了測試你的環境是否有安裝完整以及程式是否可以正常執行
```=cmd
python3 Choice_Algs.py -type ADCHA -BS 5 -FWA 20 -Thr 1 -start 0 -end 1 -unit Gbps -iter 100
```

若執行結果與下圖相同則表示可以繼續執行以下操作
![1test](https://hackmd.io/_uploads/S13NiC_U0.png)

### 2. deploy
此步驟為生成場景中SBS與FWA座標和畫50FWA 5SBS的座標圖。

```=cmd
python3 deploy.py
python3 plot_deploy.py
```

若最後畫出圖與下圖相同則表示成功
![image](https://hackmd.io/_uploads/rkK83ADIC.png)


### 3. time

此模擬對應論文中"**評估 ADCHA 與比較對象的執行時間差異**"的章節，建議執行時不要同時執行其他程式以免影響到最後執行結果，該程式會將所有場景分別執行一次並紀錄執行時間

```=cmd
bash Times.sh
```

當執行完後，請先到`output/5BS_10FWA_GoalThr6.0Mbps`資料夾中檢查日期，之後至`Execution.py`的第9行current_date參數中修改成與資料夾相同的日期。

Ex:![image](https://hackmd.io/_uploads/rJTO1k_IC.png)

修改完後就可以執行以下程式畫圖

```=cmd
python3 Execution.py
```

圖可以與論文結果圖比較，可能會根據你使用的電腦效能而有所差異，但趨勢是相同的。

![image](https://hackmd.io/_uploads/BJzzaRvUC.png)


### 4. FWA
此模擬中會執行以下場景並包含五種演算法共計25個程式同時執行。
:::success
**不同FWA數量**
* 5SBS 10FWA 600Mbps
* 5SBS 20FWA 600Mbps
* 5SBS 30FWA 600Mbps
* 5SBS 40FWA 600Mbps
* 5SBS 50FWA 600Mbps
:::
![image](https://hackmd.io/_uploads/H1OXCC_LC.png)


建議執行此實驗須使用效能優秀的電腦『32緒以上』，因為腳本設計會同時執行25個程式。


### 5. Rate



## 執行程式

```bash=
python3 Choice_Algs.py -type [使用演算法] -BS [SBS數量] -FWA [FWA數量] -Thr [吞吐量的值] -unit [吞吐量的單位] -iter [總迭代次數] -start [開始執行的次數] -end [結束執行的次數]
```

### 程式測試
此目的是測試環境是否有安裝好以及結果是否有相同

### Execution Time
在`./ExcutionTimes`資料夾中執行以下程式碼
``` command
bash Times.sh
```
執行完後會在執行的當日日期中出現
"放上圖片(當天日期資料夾)"
### 不同的參數的執行
:::warning
**建議**
在不同電腦上執行會比較快
我在執行的時候分成兩台電腦分別執行不同FWA數量和不同最小傳輸速率需求
:::


:::success
**不同最小傳輸速率需求**
* 5SBS 20FWA 600Mbps
* 5SBS 20FWA 1Gbps
* 5SBS 20FWA 2Gbps
* 5SBS 20FWA 2.5Gbps
* 5SBS 20FWA 3Gbps
:::

你需要分別在`5BS_10FWA_GOalThr1Gbps`、`5BS_20FWA_GOalThr1Gbps`、`5BS_30FWA_GOalThr1Gbps`、`5BS_40FWA_GOalThr1Gbps`、`5BS_50FWA_GOalThr1Gbps`、`5BS_20FWA_GOalThr600Mbps`、`5BS_20FWA_GOalThr1Gbps`、`5BS_20FWA_GOalThr2Gbps`、`5BS_20FWA_GOalThr2.5Gbps`、`5BS_20FWA_GOalThr3Gbps`、以上資料夾中分別執行以下程式碼

``` command
bash Start.sh
```

以上程式碼會在每種不同的參數場景中使用tmux同時執行個種不同的演算法(ADCHA、HHO、HHO_SMA、NSGAII)，以上總共會執行50個程式。


#### 執行完成後
你會看到多出一個當天日期的資料夾
"放上圖片(當天日期資料夾)"
之後就可以開始畫圖了，如果當初是在不同電腦上執行的建議把執行結果轉移到同一台電腦上(轉移電腦建議：建議先將資料夾壓縮後再統一傳送會比較快)，當轉移好後就可以開始畫圖了。




