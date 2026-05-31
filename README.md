# neural_regression_linear_model
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
<img width="1078" height="624" alt="image" src="https://github.com/user-attachments/assets/ab0c26f0-ca3f-4b2c-92d9-2d00d74f6d6e" />

## DESIGN STEPS
### STEP 1: Generate Dataset
Create input values from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve
Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line
Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions
Use the trained model to predict for a new input value .

## PROGRAM
```
Name: SYED SAIF SYED GHOUSE
Register Number: 212224230286
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Deep learning exp/deep ex 1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: I S ISHITA
# Register Number:212224220038
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
lig = NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (lig. parameters(), lr=0.001)
loss_history = []

# Name: I S ISHITA
# Register Number:212224220038
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range (epochs) :
      optimizer. zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss. backward()
      optimizer.step()
      loss_history.append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


loss_df = pd.DataFrame(loss_history, columns=["Loss"])

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
## Dataset Information
<img width="197" height="254" alt="image" src="https://github.com/user-attachments/assets/3a8ef600-2ccb-42a0-9cb7-fe4572d6395c" />

## OUTPUT
<img width="687" height="515" alt="image" src="https://github.com/user-attachments/assets/79ee01cf-d366-4127-9f02-d3b588182a0e" />

<img width="723" height="538" alt="image" src="https://github.com/user-attachments/assets/53baaacf-89d4-4647-807d-bfd767843b80" />

### Training Loss Vs Iteration Plot Best Fit line plot image
<img width="689" height="518" alt="image" src="https://github.com/user-attachments/assets/d37bfc37-590a-414b-86d0-66c13c14a9a9" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
