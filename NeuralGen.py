"""
This code represents a toy problem for the Xor gate, it uses a Artificial Neural Network
in which the parameters are set by a Genetic Algorithm, in this case, the weights are being set.
Author: Ithallo Junior Alves Guimaraes
Major: Electronics Engineering
Universidade de Brasilia
November/2015
"""
# Libraries
import numpy as np
import random as rd
import math as mt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#from pybrain.datasets import SupervisedDataSet
#import pylab as plt

# General things and parameters


# Creating the Artificial Neural Network,ANN, just an example, acording to the problem
net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
# This is the model for the vector, a.k.a. individual
model = [-1.64068868, 2.34397002, 0.05694412, -0.41128811, -1.5332341, -1.54044452, -0.20447191, -0.22938052, -1.72944555, -1.74878273, 0.82475268, -0.04566501, -2.13976602]
Pop1 = np.array([model]* 10)
# Creating data set for the Xor. [[inputA, inputB, Output], ...]
dataSet = SupervisedDataSet(2, 1)
dataSet.addSample([0, 0], [0])
dataSet.addSample([0, 1], [1])
dataSet.addSample([1, 0], [1])
dataSet.addSample([1, 1], [0])



# Function to generate the population, each individual contains the characteristic that can be changed in the ANN [hiddenlayer(0,10), bias, hiddenclass, learning rate, momentum]
def pop(pop_size,):
    population = []
    individual = []
    for i in xrange(0, pop_size):
        population. append([rd.randint(0,10), bool(rd.getrandbits(1)), rd.choice([TanhLayer, SigmoidLayer]), float(rd.randint(0, 9)/10.0), float(rd.randint(0, 9)/10.0)])
    
    return np.array(population)



# B This functions ranks
def ranking(Population, era):
        # A value to get the minimal
    minimal = 10000
    # Getting the error, here, the mean squared error of each individual, remember the dataset
    error = []
    for i in xrange(0, len(Population)):
        net = buildNetwork(2, Population[i][0] , 1, bias=Population[i][1], hiddenclass=Population[i][2])
        trainer = BackpropTrainer(net, dataSet, learningrate=Population[i][3], momentum=Population[i][4])
        # RUn till the end of the era and check the error
        midError = 0.000001
        for  tr in xrange(0, era):
                midError = trainer.train()
        error.insert(i, [midError, i])
        # Sorting the values into a new final list
    def getKey(item):
        return item[0]
    return sorted(error, key=getKey)



"""
# To do this part, my idea is to define how much (%) of the past population is going to have offspring, it receives also the percent of the population is going to mutate too. I'm thinking about setting the number of offspring per parent to be random. It should be remembered that some not fitted should have offspring too in order to keep the genetic diversity
"""
# Crossover, mutation, breeding
def breed1(Population,RANK, mutate, crossover, DyingRAte):
    # Setting the number opopulation to reproduce
    numM = int (crossover * len(Population))
    Population = Population.tolist()
    # Based on the gotten number, from here on, the said to be the best are going to mate and generate offspring, the new one will substitute the worst ones
    children = []
    mate1 = []
    mate2 = []
    for i in xrange(0, numM):
        mate1 = Population[RANK[i][1]]
        mate2 = rd.choice(Population)
        # generating the child
        children.insert(i, mate1)
        for a in xrange(0, rd.randint(1, (len(mate1) - 1))):
            children[i][a] = mate2[a]
    #reordening
    PopMid = []
    for h in xrange(0, len(RANK)):
        PopMid.append(Population[RANK[h][1]])
    
    for dl in range((len(RANK)- int(DyingRAte*len(RANK)) -1), len(RANK)):
        PopMid.pop(len(PopMid)-1)
        #print'killed', dl, "rank", len(RANK)
    #print Population, "\n\n"
    Population = PopMid + children
    print "POpulaton size", len(Population)
    #print"New pop no mutation", Population
    # Mutating
    muNum = int(mutate * len(Population))
    #print muNum
    for fi in xrange(0, muNum):
        varl = Population.index(rd.choice(Population))
        varl1 = rd.randint(0, len(Population[varl])-1)
        if varl1 == 0:
            Population[varl][varl1] = rd.randint(0,10)
        if varl1== 1:
            Population[varl][varl1] = bool(rd.getrandbits(1))
        if varl1 == 2:
            Population[varl][varl1] = rd.choice([TanhLayer, SigmoidLayer])
        else:
            Population[varl][varl1] = float(rd.randint(0, 9)/10.0)
    #print varl, varl1
    #print"New pop", Population
    return np.array(Population)



# CHANGE:PREVENT POPULATION FROM BECOMING 0
#print'\n\n\n'
# For testing purposes
era = 100
DyingRAte = 0.1
mut = 0.2
cross = 0.6
totGen = 100
POpSize =10


# create the original  population and evaluate
Population = pop(POpSize)
#print Population
#Population = Pop1
r1 = ranking(Population, era)
print r1[0], "\n\n"

for runi in xrange(0, totGen):
    Population = breed1(Population, r1, mut, cross, DyingRAte)
    r1 = ranking(Population, era)
    print runi, r1[0]
print'Best parameters:', Population[r1[0][1]]
"""
net = buildNetwork(2, Population[i][0] , 1, bias=Population[i][1], hiddenclass=Population[i][2])
print '1 XOR 1: Esperado = 0, Calculado =', net.activate([1, 1])[0]
print '1 XOR 0: Esperado = 1, Calculado =', net.activate([1, 0])[0]
print '0 XOR 1: Esperado = 1, Calculado =', net.activate([0, 1])[0]
print '0 XOR 0: Esperado = 0, Calculado =', net.activate([0, 0])[0]
#print r1[0][0]
#"""



