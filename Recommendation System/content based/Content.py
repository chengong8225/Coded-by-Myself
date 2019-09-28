# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:13:03 2019

@author: 11078
"""

import os
os.chdir('D:/Code/公众号素材/推荐系统/代码')

import pandas as pd
import math

class rec_based_content(object):
    def __init__(self, movies_path, ratings_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.get_movies_and_genres()
        self.ratings = pd.read_csv(self.ratings_path, sep = '\t')
        
    def get_movies_and_genres(self):
        '''
        计算三个重要的量，保存起来
        self.movies_and_genres - {mid:[genre]}
        self.genres_and_movies - {genre: [mid]}
        self.all_genres - 所有genre
        '''
        self.movies_and_genres = {}
        self.genres_and_movies = {}
        self.all_genres = set([])
        movies = pd.read_csv(self.movies_path, sep = '\t')
        for i in range(movies.shape[0]):
            mid, genres = movies.iloc[i][0], movies.iloc[i][-1]
            genres = genres.split('|')
            self.movies_and_genres[int(mid)] = genres
            for genre in genres:
                self.genres_and_movies.setdefault(genre, [])
                self.genres_and_movies[genre].append(mid)
            self.all_genres = self.all_genres | set(genres)

    def cal_preference(self, df):
        '''
        input:  df是uid打分过的电影， mid-rating
        output: prefer - uid对各类mid的偏好，格式为dict
                what_you_saw - 偏好的genres
        '''
        prefer = {}
        for idx in range(df.shape[0]):
            mid, rating = df.iloc[idx]
            for genre in self.movies_and_genres[mid]:
                prefer.setdefault(genre, [0, 0])
                prefer[genre][0] += rating
                prefer[genre][1] += 1
        
        all_ratings = sum(map(lambda x: x[0], prefer.values()))
        all_cnt = sum(map(lambda x: x[1], prefer.values()))
        avg = all_ratings * 1.0 / all_cnt
        
        what_you_saw = set(prefer.keys())
        for genre in self.all_genres:
            if genre not in prefer:
                prefer[genre] = 0.0
            else:
                prefer[genre] = prefer[genre][0]*1.0/prefer[genre][1] - avg
        return prefer, what_you_saw
    
    def cal_distance(self, prefer, genres):
        '''
        input ：prefer - 用户uid的偏好
                genres - 具体某个电影mid有多少个genre
        output：uid和mid的距离，这里使用余弦距离
        '''
        ratings_2 = sum(map(lambda x: x**2, prefer.values()))
        cnt_2 = len(genres)
        res = 0.0
        for g in genres:
            res += prefer[g] 
        return res*1.0 / (math.sqrt(ratings_2) * math.sqrt(cnt_2))
        
    def recommend(self, uid, K):
        ratings = self.ratings[self.ratings['uid'] == uid][['mid', 'rating']]
        what_you_saw = list(ratings['mid'])
        prefer, genres_you_saw = self.cal_preference(ratings)
        score = {}
        for g in genres_you_saw:
            for m in self.genres_and_movies[g]:
                if m in what_you_saw or m in score:
                    continue
                score[m] = self.cal_distance(prefer, self.movies_and_genres[m])

#        for m in self.movies_and_genres.keys():
#            if m in what_you_saw or m in score:
#                continue
#            score[m] = self.cal_distance(prefer, self.movies_and_genres[m])
        res = sorted(score.items(), key = lambda x: x[1], reverse = True)[:K]
        return res

if __name__ == '__main__':
    movies_path = './data/ml-1m/movies.csv'
    ratings_path = './data/ml-1m/ratings.csv'
    rbc = rec_based_content(movies_path, ratings_path)
    rec = rbc.recommend(1, 10)
    print(rec)