# Mechine Learning Offline Networkflow Detection

## 功能
- 特徵萃取
- 模型訓練
- 資料預測

## 主要操作檔案
- final/
  - featurextration/
    - raw/(存放原始封包檔案)
      - demo-ioc/
    - csv/(特徵萃取過的檔案)
      - full_data.csv
    - run.sh
  - model/
    -  lstm.h5
    -  detect.py
    -  test.csv
    -  train.csv

## 操作說明
- 模型訓練
  > detect.py -t train
- 特徵萃取
  > run.sh raw/foldername
