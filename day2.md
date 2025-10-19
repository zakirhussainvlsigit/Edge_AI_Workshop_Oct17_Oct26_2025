---
# Day2 onwards learning activities 
---

<table border="1" cellspacing="0" cellpadding="5" width="100%">
  <tr>
    <th colspan="2" align="center">Programming: Logic vs Learning</th>
  </tr>
  <tr>
    <td width="50%" align="center"><b>Traditional Logic</b></td>
    <td width="50%" align="center"><b>AI</b></td>
  </tr>
  <tr>
    <td>You write the rules</td>
    <td>You feed it examples</td>
  </tr>
  <tr>
    <td>Code is predictable</td>
    <td>It learns patterns</td>
  </tr>
  <tr>
    <td>Like RISC-V assembly: clean, minimal, direct</td>
    <td>Like a toddler learning by trial and error</td>
  </tr>
</table>

---

<img width="1264" height="686" alt="Screenshot 2025-10-19 221743" src="https://github.com/user-attachments/assets/09897948-e448-400d-9f07-6a12eac7e9dd" />

---
## Gradient Descent Unlocked - Build Your First AI Model From Scratch

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
print(dataset)

plt.scatter(dataset['Hours'],dataset['Scores'])
plt.show()

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

print(X)
print(Y)

Y=Mx+c

class Model():
  def __init__(self, learning_rate, iterations):
	  self.learning_rate=learning_rate
	  self.iterations=iterations
	 
  def predict(self,X):
	  return X.dot(self.slope)+self.const
	 
  def fit(self,X,Y):
	  self.m, self.n=X.shape
	  self.slope=np.zeros(self.n)
	  self.const=0
	  self.X=X
	  self.Y=Y
	  
	  for i in range(self.iterations):
	    self.update_weights()
	  return self
	  
  def update_weights(self):
	  Y_pred=self.predict(self.X)
	  dW = -(2*(self.X.T).dot(self.Y - Y_pred))/ self.m
	  db = - 2*np.sum(self.Y - Y_pred)/self.m
	  
	  self.slope = self.slope - self.learning_rate * dW
	  self.const = self.const - self.learning_rate * db
	  return self
	  
	  
model = Model(learning_rate=0.01, iterations=1000)
model.fit(X,Y)

Y_pred=model.predict(X)
print(Y_pred)

plt.scatter(dataset['Hours'],dataset['Scores'])
plt.scatter(X,Y_pred,color='red')
plt.show()

```

[Gradient Descent Unlocked - Build Your First AI Model From Scratch](https://colab.research.google.com/drive/1wmxUSlbn2_UM20fqcJB-5uXazHyMxu9A?usp=sharing)

