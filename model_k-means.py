#!usr/bin/env python
#_*_ coding:utf-8 _*_
import random
import math
import numpy as np


def euclideanDistance(x,y):
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(x,y)]))

#L=points,
def partition(points, k, means, d=euclideanDistance):

   thePartition = [[] for _ in means]  # list of k empty lists

   indices = range(k)


   for x in points:

      closestIndex = min(indices, key=lambda index: d(x, means[index]))#实现X与每个Y直接的求解：key=lambda index: d(x, means[index])

      thePartition[closestIndex].append(x)

   return thePartition


def mean(points):
   ''' assume the entries of the list of points are tuples;
       e.g. (3,4) or (6,3,1). '''

   n = len(points)
   return tuple(float(sum(x)) / n for x in zip(*points))  #将最开始的[[4, 1], [1, 5]] 经过处理变成[（4, 1）,（1, 5）]


def kMeans(points, k, initialMeans, d=euclideanDistance):
   oldPartition = []
   newPartition = partition(points, k, initialMeans, d)

   while oldPartition != newPartition:
      oldPartition = newPartition
      newMeans = [mean(S) for S in oldPartition]
      newPartition = partition(points, k, newMeans, d)

   return newPartition


def importData():
   f = lambda price,age,name: [float(price), float(age),name]

   with open('test_data','r',encoding='utf-8') as inputFile:
          return [f(*line.strip().split('\t')) for line in inputFile]



if __name__ == "__main__":
   L = [x[0:2] for x in importData()] # remove names


   import matplotlib.pyplot as plt

   import random
   k = 5
   partition = kMeans(L, k, random.sample(L, k))  #L是集合，K分类个数，random.sample(L, k)中心点
   color=[]
   mark = ['s','o','^','v','_','<','d','p','h','8','+','*','h','H','>']
   for i in range(k):

       print(i)
       plt.scatter(*zip(*partition[i]), c=(np.random.rand(1,3)),marker=mark[i],alpha=0.5,label=i)
   plt.legend()
   plt.show()