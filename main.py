import numpy as np
import pandas as pd
from glob import glob
from sklearn.encsemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#データの読み込み
files = sorted(glob("train_features_*.jsonl"))

for file in files:
  chunks =pd.read_json(
    file,
    lines = True,
    chunksize = 10000
  )
  for chunk in chunks:
    process(chunk)

#特徴量(100個のデータと、10個の特徴)
X = np.random.rand(100,10)
#ラベル(0が良性、1がマルウェア) 0と1からなる長さ100の配列を生成
y = np.random.randint(0, 2, 100)

#学習用データとテスト用データに分割
X_train, X_test, y_train, y_test = train_test_split(X, y,