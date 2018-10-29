import pandas as pd
import numpy as np

train = pd.read_csv('C:/Users/yutak/PycharmProjects/machine/Kaggle/Titanic/Data/train.csv')
test = pd.read_csv('C:/Users/yutak/PycharmProjects/machine/Kaggle/Titanic/Data/test.csv')
train.head()
test_shape = test.shape
train_shape = train.shape
test_stastic = test.describe()
train_stastic = train.describe()

def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns
#欠損値がデータフレームにどれだけ含まれているかカラムごとに集計
kesson_table_train = kesson_table(train)
kesson_table_test = kesson_table(test)

#データをきれいにする
# 1. 欠損データを代理データにする 2. 文字列カテゴリデータを数字に変換
'''
trainはAge, Embarked Cabinに欠損データがある
Cabinは予測モデルに使わないことにするため、AgeとEmbarkedに
代理値を入れる。
Ageについてはtrainの中のageの中央値(Median)を使うことにする。
(ここが重要なポイントでもある。通常はもっと複雑なことをする)

Embarkedは欠損データが2つある。ほかのデータを見たときに
Sが一番多かったので代理データにSを使う。
'''
train["Age"] = train["Age"].fillna(train["Age"].median())#Train[Age]のNaNにtrain[Age]の中央値を埋める
train["Embarked"] = train["Embarked"].fillna("S")
kesson_table(train)

'''
今回の予想で使う項目で文字列を値として持っているカラムは「Sex」と「Embarked」の2種類。
Sexは「male」「female」の2つのカテゴリー文字列、Embarkedはは「S」「C」「Q」の3つの文字列。
これらを数字に変換する。
'''

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train.head(10)


test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()
test.head(10)

'''
決定木を使った予測モデルで訓練
タイタニックに乗船していた客の「チケットクラス（社会経済的地位）」「性別」「年齢」「料金」のデータを元に
生存したか死亡したかを予測する
'''
from sklearn import tree


# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

my_prediction.shape
print(my_prediction)






'''
CSVの書き出し
'''
# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])
