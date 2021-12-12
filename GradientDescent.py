import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from mpl_toolkits import mplot3d
sns.set_style("darkgrid") 


class Gradient_Descent:
  def __init__(self,alpha=0.01,parameters=np.array([3,4]),steps=1000):
    self.alpha = alpha
    self.params = parameters
    self.steps = steps
    self.history= []
    self.fx_vals= []
    
  def f(self,x1,x2):
    return x1**2+x2**2+4

  def gradient(self,x1,x2):
    return np.array([2*x1,2*x2])

  def gradient_descent(self):    
    for i in range(self.steps):
      self.params = self.params - self.alpha*(self.gradient(self.params[0],self.params[1]))  
      f_val = self.f(self.params[0],self.params[1])
      self.history.append(self.params)
      self.fx_vals.append(f_val)
    return self.history,self.fx_vals


grad_desc = Gradient_Descent()
history,fx_vals = grad_desc.gradient_descent()

print("Minimum Value of the function : ",fx_vals[-1])

x1 = np.array([i[0] for i in history])
x2 = np.array([i[1] for i in history])
plt.title("Values vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Values")
plt.plot(x1,label="x1")
plt.plot(x2,label="x2")
plt.legend()
plt.show()

plt.plot(fx_vals,label="Function Value")
plt.xlabel("Epochs")
plt.ylabel("Values")
plt.legend()
plt.show()