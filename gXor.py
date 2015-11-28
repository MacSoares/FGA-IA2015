from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
import numpy as np
import random as rd
import math as mt
from Tkinter import *
import NeuralGenMSEMod as ngm
master = Tk()
#Labels for the code

Label(master, text = "Population:").grid(row=1, column=0)
Label(master, text = "Generations:").grid(row=2, column=0)
Label(master, text = "Crossover rate:").grid(row=3,column=0)
Label(master, text = "Mutation rate:").grid(row=4, column=0)

#VAriables
popu = IntVar(master)
totgen = IntVar(master)
cros = DoubleVar(master)
muta = DoubleVar(master)
# setting the inicial values
popu.set(100)
totgen.set(100)
cros.set(0.1)
muta.set(0.1)
# creating the options
optionPop = OptionMenu(master, popu, 100, 300, 500, 700, 1000, 2000, 3000).grid(row=1, column=2)
optionTotgen = OptionMenu(master, totgen, 100, 500, 750, 1000, 2000).grid(row=2, column=2)
optionCross = OptionMenu(master, cros, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0).grid(row=3, column=2)
optionMutation = OptionMenu(master, muta, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0).grid(row=4, column=2)

# For testing purposes
net1 = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
dataset = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]

#Button
def ok():
    x1=[]
    y1=[]
    Population1 = ngm.pop(net1.params, popu.get(), 0, 1)
    rank1 = ngm.ranking(Population1, dataset, net1)
    s1 = DoubleVar()
    s1.set(rank1[0][0])
    Label(master, text="Initial error:").grid(row=5, column=0)
    Label(master, textvariable = s1).grid(row=5, column=2)
    #passing the generations
    for un in xrange(0, totgen.get()):
        Population1 = ngm.breed(Population1, rank1, muta.get(), cros.get(), 0, 1, cros.get())
        rank1 = ngm.ranking(Population1, dataset, net1)
    s2 = DoubleVar()
    s2.set(rank1[0][0])
    Label(master, text="Final error:").grid(row=6, column=0)
    Label(master, textvariable = s2).grid(row=6, column=2)
    # Printing data
    net1._setParameters(Population1[rank1[0][1]])

    s4 = DoubleVar()
    s4.set(net1.activate([0, 0])[0])
    Label(master, text="0 XOR 0 = 0, final:").grid(row=7, column=0)
    Label(master, textvariable = s4).grid(row=7, column=2)

    s5 = DoubleVar()
    s5.set(net1.activate([0, 1])[0])
    Label(master, text="0 XOR 1 = 1, final:").grid(row=8, column=0)
    Label(master, textvariable = s5).grid(row=8, column=2)

    s6 = DoubleVar()
    s6.set(net1.activate([1, 0])[0])
    Label(master, text="1 XOR 0 = 1, final:").grid(row=9, column=0)
    Label(master, textvariable = s6).grid(row=9, column=2)

    s7 = DoubleVar()
    s7.set(net1.activate([1, 1])[0])
    Label(master, text="1 XOR 1 = 0, final:").grid(row=10, column=0)
    Label(master, textvariable = s7).grid(row=10, column=2)


button = Button(master, text="OK", command=ok).grid(row=1, column=3)


master.title("Genetic Algorithm")
master.geometry("400x300+400+200")
master.mainloop()
