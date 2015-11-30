"""
This code represents a set of functions to implement  the toy problem for the Xor gate, it uses an Artificial Neural Network
in which the parameters are set by a Genetic Algorithm, in this case, the weights are being set. This must be used with
the interface, as it does not print any result on the screen and works just as a set of functions.
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

# General things and parameters


# Creating the Artificial Neural Network,ANN, just an example, acording to the problem
#net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
# Creating data set for the Xor. [[inputA, inputB, Output], ...]
#dataSet = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]


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

"""
To do this part, my idea is to define how much (%) of the past population is going to have offspring, it receives also the percent of the 
population is going to mutate too. It should be remembered that some not fitted should have offspring too in order to keep the genetic diversity
"""
# Crossover, mutation, breeding
def breedND(Population,RANK, mutate, crossover, mu, sigma):
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
    # Mutating
    muNum = int(mutate * len(Population))
    for fi in xrange(0, muNum):
        varl = Population.index(rd.choice(Population))
        varl1 = rd.randint(0, len(Population[varl])-1)
        Population[varl][varl1] = rd.gauss(mu, sigma)
    return np.array(Population)

def breed(Population,RANK, mutate, crossover, mu, sigma, DyingRAte):
    # Setting the number opopulation to reproduce
    numM = int(crossover * len(Population))
    Population = Population.tolist()
    # Based on the gotten number, from here on, the said to be the best are going to mate and generate offspring, the new one will substitute the worst ones
    children = []
    mate1 = []
    mate2 = []
    for i in xrange(0, numM):
        mate1 = Population[RANK[i][1]]
        mate2 = rd.choice(Population)
        children.insert(i, mate2)
        for a in xrange(0, rd.randint(1, (len(mate1) - 1))):
            children[i][a] = mate1[a]
    #reordening
    PopMid = []
    for h in xrange(0, len(RANK)):
        PopMid.append(Population[RANK[h][1]])
    
    for dl in range((len(RANK)- int(DyingRAte*len(RANK))), len(RANK)):
        PopMid.pop(len(PopMid)-1)
    Population = PopMid + children
    # Mutating
    muNum = int(mutate * len(Population))
    #print muNum
    for fi in xrange(0, muNum):
        #protecting the past best fit from mutating
        varl = Population.index(rd.choice(Population))
        varl1 = rd.randint(0, len(Population[varl])-1)
        Population[varl][varl1] = rd.gauss(mu, sigma)
    return np.array(Population)



