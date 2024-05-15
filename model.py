# SIR Model

import pandas as pd
import matplotlib.pyplot as plt
from methods import *

import numpy as np


class SIR:
    def __init__(self, cfg):
        self.Susceptible = cfg.SIR.Susceptible
        self.Infected = cfg.SIR.Infected
        self.Resistant = cfg.SIR.Resistant
        self.betha = cfg.SIR.betha
        self.epsilon = cfg.SIR.epsilon
        self.N = cfg.SIR.N
        self.B = cfg.SIR.B
        self.b = cfg.SIR.b
        self.mu = cfg.SIR.mu
        self.P = cfg.SIR.P
        self.numIndividuals = cfg.SIR.Susceptible + cfg.SIR.Infected + cfg.SIR.Resistant
        self.results = None
        self.modelRun = False
        self.pi = cfg.SIR.pi
        self.T0 = cfg.SIR.T0
        self.T1 = cfg.SIR.T1
        self.H = cfg.SIR.H

    def run(self, method):

        def func(Y, X):
            Susceptible = Y[0]
            Infected = Y[1]
            Resistant = Y[2]
            return np.array(
                [
                    self.mu * (Susceptible + Infected +Resistant)
                    - self.betha * Susceptible * Infected
                    - self.epsilon * self.P * Susceptible
                    + (1 - self.epsilon) * self.N * Infected
                    - self.mu * Susceptible
                    + self.epsilon * self.P * Resistant
                    - (1 - self.epsilon) * self.N * Susceptible,

                    self.betha * Susceptible * Infected
                    + self.epsilon * self.P * Susceptible
                    - (1 - self.epsilon) * self.N * Infected
                    - self.mu * Infected
                    - self.b * Infected,
                    
                    self.b * Infected
                    + (1 - self.epsilon) * self.N * Susceptible
                    - self.mu * Resistant
                    - self.epsilon * self.P * Resistant,
                ]
            )

        Y0 = np.array([self.Susceptible, self.Infected, self.Resistant])

        t_method, y_method = method(func, Y0, self.T0, self.T1, self.H)
        self.results = pd.DataFrame.from_dict(
            {
                "Time": t_method,
                "Susceptible": y_method[:, 0],
                "Infected": y_method[:, 1],
                "Resistant": y_method[:, 2],
                "Dead": self.numIndividuals - y_method[:, 0] - y_method[:, 1] - y_method[:, 2]
            },
            orient="index",
        ).transpose()
        self.modelRun = True

    def plot(self,index = 1):
        if self.modelRun == False:
            print("Error: Model has not run. Please call SIR.run()")
            return
        plt.plot(self.results["Time"], self.results["Susceptible"], color="blue")
        plt.plot(self.results["Time"], self.results["Infected"], color="red")
        plt.plot(self.results["Time"], self.results["Resistant"], color="green")
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.legend(
            ["Susceptible", "Infected", "Resistant"],
            prop={"size": 10},
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        S_p = (self.Susceptible + self.Infected + self.Resistant) /(self.betha * self.Susceptible / (self.b + self.mu))
        I_p = (self.Susceptible + self.Infected + self.Resistant) * (self.mu / (self.mu + self.b)) /(1 - 1 / self.betha * self.Susceptible / (self.b + self.mu))
        R_p = (self.Susceptible + self.Infected + self.Resistant) * (self.b / (self.mu + self.b)) /(1 - 1 / self.betha * self.Susceptible / (self.b + self.mu))
        plt.title(f"ep_coef = {self.betha * self.Susceptible / (self.b + self.mu)} \nS_p = {S_p}\n I_p{I_p}\n R_p{R_p}")
        plt.savefig(r"test{index}.png")
        plt.close()
