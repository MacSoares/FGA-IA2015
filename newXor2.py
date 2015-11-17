from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from Tkinter import *
import matplotlib.pyplot as plt
import math
# from pybrain.structure import SigmoidLayer


# gerando interface
inter = Tk()
inter.title("Resultados")
inter.geometry("600x480+400+100")


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

# Definicao de variaveis a serem consideradas
max_error = 1
error = 0.0000000001
epocas = 1000

# inicializando contador de epocas
epocasPercorridas = 0


print "Pesos iniciais aleatorios: ", net.params


# definindo os pesos fixos
# fixed = [1.09463225, -0.08340289, -0.98393181, -0.45815173, -1.24558482, 0.73779261, -0.48119634,  1.58945971, 0.28771591, -1.79590746, 1.57981982, 0.09161765, 1.52277482]
# net._setParameters(fixed)
# print "\n pesos fixos:", net.params


#mostrar os pesos iniciais
Label(inter, text='\nPesos iniciais:', fg = 'red').pack()
pesi = StringVar()
pesi.set(net.params)
Label(inter, textvariable=pesi).pack()

# definir vetores para os erros e pesos
todos_erros = []
ntot = []

# Treinamento com parado por erro ou por epocas
while epocas > 0:
    error = trainer.train()
    epocas = epocas - 1
    
    todos_erros.insert(epocasPercorridas, error)
    ntot.insert(epocasPercorridas, epocasPercorridas)
    
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



#mostrando o total de epocas
epoc = Label(inter,text='Numero de epocas percorridas: ',fg = 'red').pack()
epo = IntVar()
epoc1 = Label(textvariable=epo).pack()
epo.set(epocasPercorridas)

#mostrando o erro
Label(inter, text='Erro final:',fg = 'red').pack()
er = DoubleVar()
er.set(error)
Label(inter,textvariable=er).pack()


#mostrando o 1 valor
Label(inter, text='\n1 XOR 1: Esperado = 0,Calculado:',fg = 'blue').pack()
m1 = DoubleVar()
m1.set(net.activate([1, 1])[0])
Label(inter,textvariable=m1).pack()

#mostrando o 2 valor
Label(inter, text='1 XOR 0: Esperado = 1,Calculado:',fg = 'blue').pack()
m2 = DoubleVar()
m2.set(net.activate([1, 0])[0])
Label(inter,textvariable=m2).pack()

#mostrando o 3 valor
Label(inter, text='0 XOR 1: Esperado = 1,Calculado:',fg = 'blue').pack()
m3 = DoubleVar()
m3.set(net.activate([0, 1])[0])
Label(inter,textvariable=m3).pack()

#mostrando o 4 valor
Label(inter, text=' 0 XOR 0: Esperado = 0,Calculado:',fg = 'blue').pack()
m4 = DoubleVar()
m4.set(net.activate([0, 0])[0])
Label(inter,textvariable=m4).pack()


#mostrar pesos finais
Label(inter, text='\nPesos finais:',fg = 'red').pack()
pesf= StringVar()
pesf.set(net.params)
Label(inter,textvariable=pesf).pack()


#print(todos_pesos, todos_erros, "\n")

#plotando
plt.plot(ntot, todos_erros)
plt.xlabel('Epocas')
plt.ylabel('Erro')
plt.grid(True)
plt.ylim((-1,2))
plt.show()
inter.mainloop()
