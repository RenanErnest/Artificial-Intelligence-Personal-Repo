import numpy as np
import pandas as pd
import random

class k_means:

    def __init__(self, K, trainSet):
        '''
        :param K: amount of groups desired
        :param trainSet: pandas dataframe
        '''

        self.data = trainSet.values
        self.Ks = np.zeros((K,self.data.shape[1]))
        k1 = random.randint(0, K - 1)
        self.Ks[0] = self.data[k1]
        summ = 0
        for caseValues in self.data:
            summ += self.distance(caseValues,self.Ks[0])**2
        probs = [self.distance(x,self.Ks[0])**2/summ for x in self.data]
        indexes = np.arange(len(self.data))

        print(probs,indexes)
        j = np.random.choice(indexes,p=probs,size=self.Ks.shape[0]-1,replace=False)
        for i in range(1,self.Ks.shape[0]-1):
            self.Ks[i] = self.data[j[i]]
        print(self.Ks)
        self.U = np.zeros((K,self.data.shape[0]))


    def distance(self,x1,x2):
        if type(x1) == int and type(x2) == int:
            return abs(x2-x1)
        if type(x1) != type(x2) or len(x1) != len(x2):
            return False

        '''Euclidean'''
        summ = 0
        for i in range(len(x1)):
            summ += (x1[i]-x2[i])**2
        return np.sqrt(summ)

    def train(self,error):
        '''
        :param error: an acceptable distance between the current and the previous Ks positions
        '''

        def norm(x1):
            summ = 0
            for line in x1:
                for value in line:
                    summ += abs(value)
            return summ

        Ks_prev = np.zeros((self.Ks.shape[0],self.Ks.shape[1]))
        while norm(self.Ks-Ks_prev) > error:
            Ks_prev = self.Ks.copy()

            # U updating
            for j in range(self.U.shape[1]):
                minimum_distance = float('Inf')
                minimum_index = 0
                for i in range(self.U.shape[0]):
                    self.U[i][j] = 0
                    distance = self.distance(self.Ks[i],self.data[j])
                    if distance < minimum_distance:
                        minimum_distance = distance
                        minimum_index = i
                self.U[minimum_index][j] = 1

            print(self.U)

            # Ks updating
            # Ks i get the average position between all data that belong to Ks i
            for i in range(self.Ks.shape[0]):
                summ = np.zeros((self.Ks.shape[1]))
                amt = 0
                for j in range(self.U.shape[1]):
                    summ += self.U[i][j] * self.data[j]
                    amt += self.U[i][j]
                if amt != 0:
                    self.Ks[i] = summ/amt

            x = norm(self.Ks - Ks_prev)
            print(x)

    def predict(self,data):
        data = data.values

        for caseValues in data:
            minimum_distance = float('Inf')
            minimum_index = 0
            for i in range(self.Ks.shape[0]):
                distance = self.distance(self.Ks[i], caseValues)
                if distance < minimum_distance:
                    minimum_distance = distance
                    minimum_index = i
            print('Group: ',minimum_index)


data = pd.read_csv('MLP_Data/problemAND.csv', header=None)
data = data.drop(data.columns[-1], axis=1)
km = k_means(4,data)
km.train(1)
km.predict(data)

data = pd.read_csv('MLP_Data/caracteres-limpo.csv', header=None)
data = data.drop(data.columns[-1], axis=1)
km = k_means(7,data)
km.train(1)
km.predict(data)


