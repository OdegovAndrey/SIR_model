# SIR Model

import pandas as pd
import matplotlib.pyplot as plt
from methods import *

import numpy as np

class SIR:
    def __init__(self, eons=1000, Susceptible=1500, Infected=3, Resistant=0, betha=8.3*10**(-4), epsilon=0.75,N = 1,B = 100, mu = 1/144, P = 2,b = 2,pi =10):
        '''self.eons = eons
        self.Susceptible = Susceptible
        self.Infected = Infected
        self.Resistant = Resistant
        self.rateSI = rateSI
        self.rateIR = rateIR
        self.numIndividuals = Susceptible + Infected + Resistant
        self.results = None
        self.modelRun = False'''
        self.eons = eons
        self.Susceptible = Susceptible
        self.Infected = Infected
        self.Resistant = Resistant
        self.betha = betha
        self.epsilon = epsilon
        self.N = N
        self.B = B
        self.b = b
        self.mu = mu
        self.P = P
        self.numIndividuals = Susceptible + Infected + Resistant
        self.results = None
        self.modelRun = False
        self.pi = pi


    def run(self,method):
            
        def func(Y,X):
            Susceptible = Y[0]
            Infected = Y[1]
            Resistant = Y[2]
            return np.array(
            [
                self.pi - self.betha*Susceptible*Infected - self.epsilon*self.P*Susceptible+ (1-self.epsilon)*self.N*Infected - self.mu * Susceptible  + self.epsilon * self.P*Resistant -(1- self.epsilon) * self.N * Susceptible,
                self.betha * Susceptible*Infected + self.epsilon * self.P * Susceptible - (1 - self.epsilon)*self.N*Infected - self.mu * Infected - self.b * Infected,
                self.b * Infected + (1 - self.epsilon)*self.N * Susceptible -self.mu * Resistant - self.epsilon * self.P*Resistant
            ]
            )

        T0 = 0
        T1 = 10
        H = 0.01                  #TODO: нормальные начальные условия
        Y0 = np.array([1500, 3, 0])
        L = -1


        t_method, y_method = method(func, Y0, T0, T1, H)

        #print (y_method[:,1])
        self.results = pd.DataFrame.from_dict({'Time':t_method,
            'Susceptible':y_method[:,0], 'Infected':y_method[:,1], 'Resistant':y_method[:,2]},
            orient='index').transpose()
        self.modelRun = True

    def plot(self):
        if self.modelRun == False:
            print('Error: Model has not run. Please call SIR.run()')
            return
        plt.plot(self.results['Time'], self.results['Susceptible'], color='blue')
        plt.plot(self.results['Time'], self.results['Infected'], color='red')
        plt.plot(self.results['Time'], self.results['Resistant'], color='green')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend(['Susceptible','Infected','Resistant'], prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fancybox=True, shadow=True)
        plt.title(r'beta = -, gamma = -')
        plt.savefig('test.png')
        plt.close()
        
