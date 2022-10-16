#coding:utf-8
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance as dis

"""
参考URL
[1] 蟻コロニー最適化 - Wikipedia https://ja.wikipedia.org/wiki/蟻コロニー最適化
[2] 任意の確率密度分布に従う乱数の生成（von Neumannの棄却法） | Pacocat's Life http://pacocat.com/?p=596
[3] Algorithms with Python / 巡回セールスマン問題 http://www.geocities.jp/m_hiroi/light/pyalgo64.html

https://tadaoyamaoka.hatenablog.com/entry/2017/02/19/172744

"""


class TSP:
    def __init__(self, path=None, alpha=1.0, beta=1.0, Q=1.0, vanish_ratio=0.95):
        """ 初期化を行う関数 """
        self.alpha = alpha                  # フェロモンの優先度
        self.beta = beta                    # ヒューリスティック情報(距離)の優先度
        self.Q = Q                          # フェロモン変化量の係数
        self.vanish_ratio = vanish_ratio    # 蒸発率
        if path is not None:
            self.set_loc(np.array(pd.read_csv(path)))
        # End if
    # End def

    def set_loc(self, locations):
        """ 位置座標を設定する関数 """
        self.loc = locations                            # x,y座標
        self.n_data = len(self.loc)                     # データ数
        self.dist = dis.squareform(dis.pdist(self.loc))     # 距離の表を作成
        self.weight = np.random.random_sample((self.n_data, self.n_data)) + 1.0    # フェロモンの量
        self.result = np.arange(self.n_data)        # もっともよかった順序を保存する
    # End def

    def cost(self, order):
        """ 指定された順序のコスト計算関数 """
        n_order = len(order)
        return np.sum([self.dist[order[i], order[(i + 1) % n_order]] for i in np.arange(n_order)])
        # order2 = np.r_[order[1:], order[0]]
        # return np.sum(tsp.dist[order, order2])
    # End def

    def plot(self, order=None):
        """ 指定された順序でプロットする関数 """
        if order is None:
            plt.plot(self.loc[:, 0], self.loc[:, 1])
        else:
            plt.plot(self.loc[order, 0], self.loc[order, 1])
        # End if
        plt.show()
    # End def

    def solve(self, n_agent=1000):
        """ 巡回セールスマン問題を蟻コロニー最適化で解く """
        order = np.zeros(self.n_data, np.int)           # 巡回経路
        delta = np.zeros((self.n_data, self.n_data))    # フェロモン変化量

        for k in range(n_agent):
            city = np.arange(self.n_data)
            now_city = np.random.randint(self.n_data)       # 現在居る都市番号

            city = city[city != now_city]
            order[0] = now_city

            for j in range(1, self.n_data):
                upper = np.power(self.weight[now_city, city], self.alpha) * np.power(self.dist[now_city, city], -self.beta)

                evaluation = upper / np.sum(upper)              # 評価関数
                percentage = evaluation / np.sum(evaluation)       # 移動確率

                index = self.random_index2(percentage)          # 移動先の要素番号取得

                # 状態の更新
                now_city = city[index]
                city = city[city != now_city]
                order[j] = now_city
            # End for
            L = self.cost(order)        # 経路のコストを計算

            # フェロモンの変化量を計算
            delta[:, :] = 0.0
            c = self.Q / L
            for j in range(self.n_data - 1):
                delta[order[j], order[j + 1]] = c
                delta[order[j + 1], order[j]] = c
            # End for

            # フェロモン更新
            self.weight *= self.vanish_ratio
            self.weight += delta

            # 今までで最も良ければ結果を更新
            if self.cost(self.result) > L:
                self.result = order.copy()
            # End if

            # デバッグ用
            print("Agent ... %d,\t Now Cost %lf,\t Best Cost ... %lf" % (k, L, self.cost(self.result)))
        # End for
        return self.result
    # End def

    def save(self, out_path):
        """ 最もコストが低かった順序で保存する関数 """
        points = self.loc[self.result]
        f = open(out_path, "w")
        f.write("x,y\n")
        for i in range(len(points)):
            f.write(str(points[i, 0]) + "," + str(points[i, 1]) + "\n")
        # End for
        f.close()
    # Ebd def

    def random_index(self, percentage):
        """ 任意の確率分布に従って乱数を生成する関数 """
        n_percentage = len(percentage)
        arg = np.argsort(percentage)
        while True:
            index = np.random.randint(n_percentage)
            y = np.random.random()
            if y < percentage[index]:
                return index
            # Ebd if
        # End while
    # End def

    def random_index2(self, percentage):
        """ 精度低めで任意の確率分布に従って乱数を生成する関数 """
        n_percentage = len(percentage)
        arg = np.argsort(percentage)[::-1]
        n_arg = min(n_percentage, 10)
        percentage = percentage / np.sum(percentage[arg])

        while True:
            index = np.random.randint(n_arg)
            y = np.random.random()
            if y < percentage[arg[index]]:
                return arg[index]
            # End if
        # End while
    # End def


def save_edge_points(img_path, out_path):
    # 画像を読み込んでエッジ処理
    img = cv2.imread(img_path)
    edge = cv2.Canny(img, 100, 200)

    # エッジになっているx,y座標を取り出す
    h, w = edge.shape
    x = np.arange(w)
    y = np.arange(h)

    X, Y = np.meshgrid(x, y)

    # 255になっている部分がエッジ部分
    X_true = X[edge > 128]
    Y_true = Y[edge > 128]

    # エッジの点になっている座標が入っている
    index = np.array([X_true, Y_true]).T

    # 保存
    f = open(out_path, "w")
    f.write("x,y\n")
    for i in range(len(index)):
        f.write(str(index[i, 0]) + "," + str(index[i, 1]) + "\n")
    # End for
    f.close()
# End for


if __name__ == "__main__":
    # エッジを検出し保存．img.pngが対象の画像
    save_edge_points("img.png", "edge_points.csv")

    # TSPで巡回セールスマン問題として一筆書きの手順を計算・保存
    tsp = TSP(path="edge_points.csv", alpha=1.0, beta=16.0, Q=1.0e3, vanish_ratio=0.8)
    tsp.solve(100)
    tsp.save("best_order.csv")
    tsp.plot(tsp.result)
