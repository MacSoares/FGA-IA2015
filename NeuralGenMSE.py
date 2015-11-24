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
#from pybrain.datasets import SupervisedDataSet
#import pylab as plt

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

def ranking1(Population, dataSet, net):
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

def ranking2(Population, dataSet, net):
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
def breed(Population,RANK, mutate, crossover, mu, sigma):
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
        children.insert(i, mate2)
        for a in xrange(0, rd.randint(1, (len(mate1) - 1))):
            children[i][a] = mate1[a]
    # Killing the unfit and putting the children on
    for g in xrange(0, len(children)):
        Population[RANK[len(Population) - len(children) + g][1]] = children[g]
    #print"New pop no mutation", Population
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

def breed1(Population,RANK, mutate, crossover, mu, sigma, DyingRAte):
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
        children.insert(i, mate2)
        for a in xrange(0, rd.randint(1, (len(mate1) - 1))):
            children[i][a] = mate1[a]
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
        Population[varl][varl1] = rd.gauss(mu, sigma)
    #print varl, varl1
    #print"New pop", Population
    return np.array(Population)


# CHANGE:PREVENT POPULATION FROM BECOMING 0

#print'\n\n\n'
# For testing purposes
l1 = 0
l2 = 1
DyingRAte = 0.5
mut = 0.3
cross = 0.5
totGen = 300
POpSize =1000


# create the original  population and evaluate
Population = pop(net.params, POpSize, l1, l2)
#Population = Pop1
r1 = ranking(Population, dataSet, net)
print r1[0], "\n\n"

for runi in xrange(0, totGen):
    Population = breed1(Population, r1, mut, cross, l1, l2, DyingRAte)
    r1 = ranking(Population, dataSet, net)
    print runi, r1[0]

    net._setParameters(Population[r1[0][1]])
    print '1 XOR 1: Esperado = 0, Calculado =', net.activate([1, 1])[0]
    print '1 XOR 0: Esperado = 1, Calculado =', net.activate([1, 0])[0]
    print '0 XOR 1: Esperado = 1, Calculado =', net.activate([0, 1])[0]
    print '0 XOR 0: Esperado = 0, Calculado =', net.activate([0, 0])[0]
#print r1[0][0]
#"""


#Unused pieces of code
"""
    # Looking randomically for the minimal
    if error[i] < minimal:
    minimal = error[i]
    pos = i
    #print minimal
    print'minimal error:', minimal, pos
    net._setParameters(Population[pos])
    plt.plot(x1, error)
    plt.show()
    # Showing results
    print'\n\n Parameters:\n', net.params
    #print'Error:', np.array(error)
"""


