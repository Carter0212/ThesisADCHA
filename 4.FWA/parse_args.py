import argparse

def argsParser(replaceCommandStr=''):
    parser = argparse.ArgumentParser()
    ##使用資料夾來讀取分布才要用
    parser.add_argument("--NumberOfBS",
                        "-BS",
                        type=int,
                        default='normal', 
                        required=True,
                        help="decide the number of BS, default is 'normal'")
    ## 決定Domain Proxy多久改變一次使用的channel
    parser.add_argument("--NumberOfFWA",
                        "-FWA",
                        type=int,
                        default='0', 
                        required=True,
                        help="decide the number of FWA, default is '0'")
    ## 檔案要存在哪裡
    parser.add_argument("--GoalThr",
                        "-Thr",
                        type=float,
                        default='',
                        required=True,
                        help="decide the GoalThr, default is 'None'")
    parser.add_argument("--unit",
                        "-unit",
                        type=str,
                        default='',
                        required=True,
                        help="decide the unit of GoalThr, default is 'None'")
    parser.add_argument("--iteration",
                        "-iter",
                        type=int,
                        default=10,
                        required=True,
                        help="determine the number of iterations, default is 10")
    parser.add_argument("--type",
                        "-type",
                        type=str,
                        default='None',
                        required=False,
                        help="choose the type of algorithm you want to run")
    parser.add_argument("--Start_time",
                        "-start",
                        type=int,
                        default=0,
                        required=False,
                        help="choose the start time of the algorithm")
    parser.add_argument("--End_time",
                        "-end",
                        type=int,
                        default=1,
                        required=False,
                        help="choose the end time of the algorithm")
    
    if replaceCommandStr != '':
        print(replaceCommandStr.split(' '))
        return parser.parse_args(replaceCommandStr.split(' '))
    args = parser.parse_args()
    return args
