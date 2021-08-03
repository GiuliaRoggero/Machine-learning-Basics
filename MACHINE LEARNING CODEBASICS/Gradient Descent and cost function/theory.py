import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#the gradient descend ovviamente considero solo il caso di una features per avere un idea
#how to derive the equation of the regression line
#given the output and input
#given y and the feature x

#how do we know which line is the best regression line that fits the data??

#we minimaze the error (1/n)*(sum(dai= 1 a i=n) delta(i)^2) n= numero di dati reali  che ho nel mio dataset
#delta(i) distanza del punto dalla retta di regressione
#la retta che avrà il minimo errore sarà la nostra retta di regressione

#l'errore, cioè il mean square error sarà:
    
# mse = 1/n * sum(i=1 a n)(y_i - y_predicted)^2    actual data point - predicted datapoint

#the mse is also called the cost funtion 

#cost function = 1/n * sum(i=1 a n)(y_i -(m*x_i +b))^2 , y_predicted is replaced with the equation

#THE GRADIENT DESCEND IS AN ALGORITHM THAT FINDS THE BEST FIT LINE FOR GIVEN TRAINING DATA SET 

#star with some value of m and b  usually = 0 and calculate the cost 
#poi cambio m e b ci un certo valore iterato (baby step) "che trovo poi"
#e trovo quale ha il costo minore il punto minimo graficamente 

#il passo deve essere piuttosto piccolo percheè altrimenti posso superare il minimo
#praticamente il costo (mse) è sull'asse z mentre b e m sono gli assi x e y, e devo trovare 
#il minimo della funzione, come in analisi 2

#piu mi avvicino al minimo piu mi avvicino a coefficente =0

#il passo è il learning rate

#calcolo quindi la slope rispetto a m e rispetto a b (siamo in un piano tridimensionale) della funzione Mse

#rispetto a m -> (2/n)*sum(i=1 a n) ((-x_i)y_i - (m*x_i +b))

#rispetto a b -> (2/n)*sum(i=1 a n) (-(y_i -(m*x_i + b)))

#le do per scontate, hanno la direzione

#il nostro step è il learning rate

# m = m - learningrate * derivatarispettoam

# b = b - learningrate * derivatarispettoab

#ovviamente ho un valore iniziale di m e b

def gradient_descent(x,y): #start with some parameters
         
      m_curr = b_curr=0
      iterations=10000
      n=len(x)
      learning_rate=0.08 #poi lo metto a posto se serve per migliorare l'algoritmo
      for i in range(iterations):
          y_predicted = m_curr*x + b_curr
          cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
          md= -(2/n)*sum(x*(y-y_predicted))
          bd = -(2/n)*sum(y-y_predicted)
          m_curr= m_curr - md*learning_rate
          b_curr= b_curr - bd*learning_rate
          print("m {},b {},cost {}, iteration {}".format(m_curr,b_curr,cost,i))
    
    
    
x = np.array([1,2,3,4,5])
y= np.array([5,7,9,11,13])

gradient_descent(x,y)


































