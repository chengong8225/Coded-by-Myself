# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:02:25 2019

@author: Charlie
"""

import random
from collections import defaultdict
from operator import itemgetter
import math

class ItemCF(object):
    def __init__(self, rec_movies=10):
        # 找出相近的10个物品，推荐最好的10个物品
        self.n_rec_movies = rec_movies
        # self.n_sim_items = sim_items

        # 训练集和测试集的划分
        self.train = {}
        self.test = {}

        # 物品相似度矩阵
        self.item_sim_matrix = defaultdict(dict)

    # 读取文件，返回文件的每一行
    def load_file(self, path):
        first_line = True
        with open(path) as file:
            for line in file.readlines():
                if first_line:
                    first_line = False
                    continue
                yield line.strip()

    # 分训练集和测试集， 数据格式   {用户：{用户：评分， ...}}
    def get_train_and_test(self, path, ratio=0.8):
        item = set()
        train_size, test_size = 0, 0
        for line in self.load_file(path):
            uid, mid, rating, _ = line.split(',')
            item.add(mid)
            if random.random() < ratio:
                train_size += 1
                self.train.setdefault(uid, {})
                self.train[uid][mid] = float(rating)
            else:
                test_size += 1
                self.test.setdefault(uid, {})
                self.test[uid][mid] = float(rating)
        print('Train Size: %d, Test Size: %d'%(train_size, test_size))
        print('Item Size: %d'%(len(item)))

    def cal_item_sim_matrix(self):
        '''
        计算电影间的相似度，这里采用了余弦距离
        '''
        movie_score = defaultdict(int)
        movie_cross_score = defaultdict(dict)
        for uid, mid_rating in self.train.items():
            for mid_1, rating_1 in mid_rating.items():
                movie_score[mid_1] += rating_1 ** 2
                for mid_2, rating_2 in mid_rating.items():
                    if mid_1 == mid_2:
                        continue
                    movie_cross_score[mid_1].setdefault(mid_2, 0)
                    movie_cross_score[mid_1][mid_2] += rating_1 * rating_2
        
        # max_val= 0
        for mid_1, related_items in movie_cross_score.items():
            for mid_2, score in related_items.items():
                self.item_sim_matrix[mid_1][mid_2] = score * 1.0 / math.sqrt(movie_score[mid_1] * movie_score[mid_2])
                # max_val = max(max_val, self.item_sim_matrix[mid_1][mid_2])
                
        print('get items similarity matrix!')
        print('matrix size: {}'.format(len(self.item_sim_matrix)))
        # print('max_val:{}'.format(max_val))
        
    def get_favorite_movies(self, uid, n_favorite = 5):
        return dict(sorted(self.train[uid].items(), key=itemgetter(1), reverse=True)[:n_favorite])

    def get_recommended_movies(self, uid, n_favorite = 5):
        favorite_movies = self.get_favorite_movies(uid, n_favorite)
        recommend_movies = {}
        for movie in favorite_movies:
            sim_movies = sorted(self.item_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)
            for mid, sim_score in sim_movies:
                if mid not in favorite_movies:
                    recommend_movies.setdefault(mid, [0, 0])
                    recommend_movies[mid][0] += sim_score * favorite_movies[movie]
                    recommend_movies[mid][1] += sim_score
        recommend_movies = sorted(recommend_movies.items(), key = lambda item: item[1][0]/ item[1][1], reverse=True)[:self.n_rec_movies]
        return [i[0] for i in recommend_movies]       

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
    a = ItemCF()
    a.get_train_and_test('C:/Users/11078/Desktop/python代码/协同过滤/ml-latest-small/ratings.csv')
    a.cal_item_sim_matrix()
    print(a.get_recommended_movies('1'))
    a.cal_metrics()
