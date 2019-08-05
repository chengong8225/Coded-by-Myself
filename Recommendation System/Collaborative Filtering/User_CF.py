# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:02:25 2019

@author: Charlie
"""

import random
from collections import defaultdict
from operator import itemgetter
import math

class UserCF(object):
    def __init__(self, rec_movies=10, sim_users=10):
        # 找出相近的10个用户，推荐最好的10个物品
        self.n_rec_movies = rec_movies
        self.n_sim_users = sim_users

        # 训练集和测试集的划分
        self.train = {}
        self.test = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}

    # 读取文件，返回文件的每一行
    def load_file(self, path):
        first_line = True
        with open(path) as file:
            for line in file.readlines():
                if first_line:
                    first_line = False
                    continue
                yield line.strip()

    # 分训练集和测试集
    def get_train_and_test(self, path, ratio=0.8):
        train_size, test_size = 0, 0
        for line in self.load_file(path):
            uid, mid, rating, _ = line.split(',')
            if random.random() < ratio:
                train_size += 1
                self.train.setdefault(uid, {})
                self.train[uid][mid] = float(rating)
            else:
                test_size += 1
                self.test.setdefault(uid, {})
                self.test[uid][mid] = float(rating)
        print('Train Size: %d, Test Size: %d'%(train_size, test_size))
        print('User Size: %d'%(len(self.train)))



    def cal_user_sim_matrix(self):
        # 创建电影-用户的嵌套字典
        movie_user = defaultdict(set)
        for uid, mid_rating in self.train.items():
            for movie in mid_rating:
                movie_user[movie].add(uid)

        self.movie_num = len(movie_user)

        for movie, user_set in movie_user.items():
            for u in user_set:
                for v in user_set:
                    if u == v:
                        continue

                    # 得到的user_sim_matrix是u和v的共现次数
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1

        for u, other_users in self.user_sim_matrix.items():
            for v, num in other_users.items():
                self.user_sim_matrix[u][v] = num *1.0 /math.sqrt(len(self.train[u]) * len(self.train[v]))

        print('get users similarity matrix!')

    # 针对已经有的相似度矩阵，开始做推荐
    def get_recommended_movies(self, uid):
        # 给出该用户已经看过的电影
        watched_movies = set()
        for movie, _ in self.train[uid].items():
            watched_movies.add(movie)

        # 别人看过的电影，自己没看过的电影，以（相关性，评分）的形式保存数据
        movies_rating = defaultdict(list)
        for user, sim in sorted(self.user_sim_matrix[uid].items(), key=itemgetter(1), reverse=True)[:self.n_sim_users]:
            for movie, rating in self.train[user].items():
                if movie not in watched_movies:
                    movies_rating[movie].append((sim, rating))

        # print(movies_rating)

        res = []
        for movie, item in movies_rating.items():
            score = 0
            sim_sum = 0
            for sim, rating in item:
                score += sim*rating
                sim_sum += sim
            res.append((movie, score*1.0/sim_sum))
        res = sorted(res, key=lambda x:x[1], reverse=True)
        return [i[0] for i in res[:self.n_rec_movies]]
        # return res[:self.n_rec_movies]

    # 评估推荐的效果：准确率，召回率，F1-score， 覆盖率
    def cal_metrics(self):
        rec_right = 0
        test_size = 0
        rec_size = 0
        all_movies = set()

        for uid, _ in self.train.items():
            test_data = self.test.get(uid, {})
            rec_movies = self.get_recommended_movies(uid)

            if not test_data:
                continue

            for movie in rec_movies:
                if movie in test_data:
                    rec_right += 1
                all_movies.add(movie)

            test_size += len(test_data)
            rec_size += len(rec_movies)

        p, r, c = rec_right*1.0/test_size, rec_right*1.0/rec_size, rec_right*1.0/len(all_movies)
        print('Precision:%.4f \n Recall:%.4f \n F1-score:%.4f \n Coverage:%.4f' \
              %(p, r, 2*p*r/(p+r), c))

if __name__ == '__main__':
    a = UserCF()
    a.get_train_and_test('C:/Users/11078/Desktop/python代码/协同过滤/ml-latest-small/ratings.csv')
    a.cal_user_sim_matrix()
    res = a.get_recommended_movies('1')
    print(sorted(res))
    a.cal_metrics()

