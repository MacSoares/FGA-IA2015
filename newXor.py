from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
# from pybrain.structure import SigmoidLayer

# Criacao da rede
# primeiros tres parametros sao numeros de camadas In-Hidden-Out
net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)

# Criacao dos dados de treino
ds = SupervisedDataSet(2, 1)
ds.addSample([1, 1], [0])
ds.addSample([0, 0], [0])
ds.addSample([0, 1], [1])
ds.addSample([1, 0], [1])

# Criacao do backdropTrainer
trainer = BackpropTrainer(net, ds, learningrate=0.1, momentum=0.9)

# Definicao de variaveis a serem concideradas
max_error = 1
error = 0.1
epocas = 1000

# inicializando contador de epocas
epocasPercorridas = 0

print "\n\nPesos iniciais: ", net.params

# Treinamento com parado por erro ou por epocas
while epocas > 0:
    error = trainer.train()
    epocas = epocas - 1
    epocasPercorridas = epocasPercorridas + 1

    if error == 0:
        break

# Apresentacao dos resultados baseados na tabela da porta logica XOR
print "\n\nPesos finais: ", net.params
print "\nErro final: ", error

print "\n\nTotal de epocas percorridas: ", epocasPercorridas

print '\n\n1 XOR 1: Esperado = 0, Calculado = ', net.activate([1, 1])[0]
print '1 XOR 0: Esperado = 1, Calculado =', net.activate([1, 0])[0]
print '0 XOR 1: Esperado = 1, Calculado =', net.activate([0, 1])[0]
print '0 XOR 0: Esperado = 0, Calculado =', net.activate([0, 0])[0]
