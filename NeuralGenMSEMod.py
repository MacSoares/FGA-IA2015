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
#from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
#from pybrain.datasets import dfSupervisedDataSet
import pylab as plt

# General things and parameters


# Creating the Artificial Neural Network,ANN, just an example, acording to the problem
net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
# This is the model for the vector, a.k.a. individual
model = [-1.64068868, 2.34397002, 0.05694412, -0.41128811, -1.5332341, -1.54044452, -0.20447191, -0.22938052, -1.72944555, -1.74878273, 0.82475268, -0.04566501, -2.13976602]
Pop1 = np.array([model]* 10)
# Creating data set for the Xor. [[inputA, inputB, Output], ...]
dataSet = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]



# Function to generate the population,  netp(the list, array of weights, an example), population size, limit1 limit2 for random. It returns a array with the population
def pop(netp, pop_size, mu, sigma):
    population = []
    individual = []
    for i in xrange(0, pop_size):
        for j in xrange(0, len(netp)):
            individual.insert(j, rd.gauss(mu,sigma))
        population.insert(i, individual)
        individual = []
    return np.array(population)
    



# B This functions ranks the population based on the Fitness (error, in this case) based on the Artificial Neural Network, ANN, the closer it is to 0 the better it is, This functons returns the values of the error, already sorted, and the number of the position of the individual [[a, b], ...]. The input is given by the population and a data set to which it has to be compared and the  generated ANN.
def ranking(Population, dataSet, net):
        # A value to get the minimal
    minimal = 10000
    # Getting the error, here, the mean squared error of each individual, remember the dataset
    error = []
    for i in xrange(0, len(Population)):
        net._setParameters(Population[i])
        midError = 0 
        for x in dataSet:
            midError += (pow((net.activate([x[0], x[1]])[0] - x[2]), 2))
        error.insert(i, [(midError/len(dataSet)), i])
    # Sorting the values into a new final list
    def getKey(item):
        return item[0]
    return sorted(error, key=getKey)

def rankingAtan(Population, dataSet, net):
    # A value to get the minimal
    minimal = 100
    # Getting the error, here, the mean squared error of each individual, remember the dataset
    error = []
    ierror = []
    x1 = [] # to the plot
    for i in xrange(0, len(Population)):
        x1.insert(i, i) # to the plot
        net._setParameters(Population[i])
        midError = 0
        for x in dataSet:
            midError += pow(mt.atan((net.activate([x[0], x[1]])[0] - x[2])), 2)

        print midError
        error.append([(midError/len(dataSet)), i])
    #print'not sorted yet', error
    # Sorting the values into a new final list
    def getKey(item):
        return item[0]
    return sorted(error, key=getKey)

def rankingSMSE(Population, dataSet, net):
    # A value to get the minimal
    minimal = 100
    # Getting the error, here, the mean squared error of each individual, remember the dataset
    error = []
    ierror = []
    x1 = [] # to the plot
    for i in xrange(0, len(Population)):
        x1.insert(i, i) # to the plot
        net._setParameters(Population[i])
        midError = 0
        for x in dataSet:
            midError +=mt.sqrt(pow((net.activate([x[0], x[1]])[0] - x[2]), 2))
        
        print midError
        error.append([(midError/len(dataSet)), i])
    #print'not sorted yet', error
    # Sorting the values into a new final list
    def getKey(item):
        return item[0]
    return sorted(error, key=getKey)



"""
To do this part, my idea is to define how much (%) of the past population is going to have offspring, it receives also the percent of the population is going to mutate too. I'm thinking about setting the number of offspring per parent to be random. It should be remembered that some not fitted should have offspring too in order to keep the genetic diversity
"""
# Crossover, mutation, breeding
def breedND(Population,RANK, mutate, crossover, mu, sigma):
    # Setting the number opopulation to reproduce
    numM = int (crossover * len(Population))
    Population = Population.tolist()
    #print"Pop", Population
    #return numM
    # Based on the gotten number, from here on, the said to be the best are going to mate and generate offspring, the new one will substitute the worst ones
    children = []
    mate1 = []
    mate2 = []
    for i in xrange(0, numM):
        mate1 = Population[RANK[i][1]]
        #print"mate1", mate1
        mate2 = rd.choice(Population)
        #print"mate2", mate2
        # generating the child
        children.insert(i, rd.choice([mate1, mate2]))
        for a in xrange(0, rd.randint(1, (len(mate1) - 1))):
            if children[i] == mate1:
                children[i][a] = mate2[a]
            else:
                children[i][a] = mate1[a]
    #reordening
    PopMid = []
    for h in xrange(0, len(RANK)):
        PopMid.append(Population[RANK[h][1]])

    for dl in range(((len(RANK)- len(children))), len(RANK)):
        PopMid.pop(len(PopMid)-1)
    Population = PopMid + children
    #print "POpulaton size", len(Population)
    # Mutating
    muNum = int(mutate * len(Population))
    #print muNum
    for fi in xrange(0, muNum):
        varl = Population.index(rd.choice(Population))
        varl1 = rd.randint(0, len(Population[varl])-1)
        Population[varl][varl1] = rd.gauss(mu, sigma)
    #print varl, varl1
    #print"New pop", Population
    return np.array(Population)

def breed(Population,RANK, mutate, crossover, mu, sigma, DyingRAte):
    # Setting the number opopulation to reproduce
    numM = int(crossover * len(Population))
    Population = Population.tolist()
    #print"Pop", Population
    #return numM
    # Based on the gotten number, from here on, the said to be the best are going to mate and generate offspring, the new one will substitute the worst ones
    children = []
    mate1 = []
    mate2 = []
    for i in xrange(0, numM):
        mate1 = Population[RANK[i][1]]
        #print"mate1", mate1
        mate2 = rd.choice(Population)
        #print"mate2", mate2
        # generating the child
        children.insert(i, mate2)
        for a in xrange(0, rd.randint(1, (len(mate1) - 1))):
            children[i][a] = mate1[a]
    #reordening
    PopMid = []
    for h in xrange(0, len(RANK)):
        PopMid.append(Population[RANK[h][1]])
    
    for dl in range((len(RANK)- int(DyingRAte*len(RANK))), len(RANK)):
        PopMid.pop(len(PopMid)-1)
        #print'killed', dl, "rank", len(RANK)
    #print Population, "\n\n"
    Population = PopMid + children
    #print"New pop no mutation", Population
    # Mutating
    muNum = int(mutate * len(Population))
    #print muNum
    for fi in xrange(0, muNum):
        #protecting the past best fit from mutating
        varl = Population.index(rd.choice(Population))
        varl1 = rd.randint(0, len(Population[varl])-1)
        Population[varl][varl1] = rd.gauss(mu, sigma)
    #print varl, varl1
    #print"New pop", Population
    return np.array(Population)



