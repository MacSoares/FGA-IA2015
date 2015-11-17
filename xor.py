from pybrain.structure import RecurrentNetwork, FullConnection
from pybrain.structure import LinearLayer
# from pybrain.structure import SigmoidLayer
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


# Define network structure

network = RecurrentNetwork(name="XOR")

inputLayer = LinearLayer(2, name="Input")
hiddenLayer = TanhLayer(3, name="Hidden")
outputLayer = LinearLayer(1, name="Output")

network.addInputModule(inputLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outputLayer)

c1 = FullConnection(inputLayer, hiddenLayer, name="Input_to_Hidden")
c2 = FullConnection(hiddenLayer, outputLayer, name="Hidden_to_Output")
c3 = FullConnection(hiddenLayer, hiddenLayer, name="Recurrent_Connection")

network.addConnection(c1)
network.addRecurrentConnection(c3)
network.addConnection(c2)

network.sortModules()

# Add a data set
ds = SupervisedDataSet(2, 1)


ds.addSample([1, 1], [0])
ds.addSample([0, 0], [0])
ds.addSample([0, 1], [1])
ds.addSample([1, 0], [1])

# Train the network
trainer = BackpropTrainer(network, ds, learningrate=0.1, momentum=0.9, verbose=True)

# print network

print "\nPesos iniciais: ", network.params

max_error = 1
error, epocas = 5, 1000
epocasPercorridas = 0

# Train
while epocas > 0:
    error = trainer.train()
    epocas = epocas - 1
    epocasPercorridas = epocasPercorridas + 1

    if error == 0:
        break

# print "\n Treinando ate a convergencia. . ."

# trainer.trainUntilConvergence()

# print "\n\nRNA treinada ate a convergencia!"

print "\n\nPesos finais: ", network.params
print "\nErro final: ", error

print "\n\nTotal de epocas percorridas: ", epocasPercorridas

# Test data

print '\n\n1 XOR 1: Esperado = 0, Calculado = ', network.activate([1, 1])[0]
print '1 XOR 0: Esperado = 1, Calculado =', network.activate([1, 0])[0]
print '0 XOR 1: Esperado = 1, Calculado =', network.activate([0, 1])[0]
print '0 XOR 0: Esperado = 0, Calculado =', network.activate([0, 0])[0]
