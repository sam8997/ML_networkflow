# Mechine Learning Offline Networkflow Detection

## 簡述
- 透過已有的網路封包資料集，經過特徵萃取後對模型進行訓練，以達到對於離線網路流中是否有惡意流量之預測
## 功能
- 特徵萃取
- 模型訓練
- 資料預測

## 主要操作檔案
- final/
  - feature_extration/
    - raw/(存放原始封包檔案)
      - xxx/
        - xxx.pcap
        - IOC.txt(有問題的特徵)
    - csv/(特徵萃取過的檔案)
      - full_data.csv
    - run.sh

  - model/
    -  lstm.h5(模型參數)
    -  detect.py
    -  test.csv
    -  train.csv

## 操作說明
- 模型訓練 (model/)
  > python3 detect.py -t train
- 特徵萃取(feature_rxtration/)
  > tmux 
  > run.sh raw/foldername/

## 環境配置
- wsl
  > wsl --import Ubuntu {目的資料夾} {wsl.tar 路徑}
  - pipenv 
  - tmux
- docker
  - broplatform:4.1.0
- Nvidia driver (加速用)
  - cuda
