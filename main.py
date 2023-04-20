import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('AI/Linear Regression/student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = 'G3'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
y = y.reshape(395, 1)

x = torch.from_numpy(x.values.astype(np.float32))
y = torch.from_numpy(y.values.astype(np.float32))

print(x, y)
n_samples, n_features = x.shape
input_size = n_features
output_size = 1
model = torch.nn.Linear(input_size, output_size)

learning_rate = 0.10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 395
for epoch in range(num_epochs):
    y_predicted = model(x)
    loss = criterion(y_predicted, y[epoch])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item()}')
predicted = model(x).detach().numpy()
plt.plot(x, y, 'ro')
plt.plot(x, predicted, 'b')
plt.show()
'''''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)
inputs = inputs.type(torch.float32)
targets = targets.type(torch.float32)

print(inputs)
print(targets)
class LinearRegression(torch.nn.Module):
    def __init__(self) -> None:
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(5, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinearRegression()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(395):
    pred_y = model(inputs)
    loss = criterion(pred_y, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'epoch {epoch}, loss {loss.item()}')
'''''