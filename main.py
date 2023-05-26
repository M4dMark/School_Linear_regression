import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('student-mat.csv', sep=';')

    input_cols = input('Enter features from the dataset, you want to use to predict by (feature,feature): ').split(',')
    target_cols = input('Enter the feature you want to predict (feature): ')
    x = data[input_cols].values
    y = data[target_cols].values

    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    input_size = x.shape[1]
    hidden_size = 32
    output_size = y.shape[1]

    model = nn.Sequential(nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size)
                        )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    batch_size = 32

    for epoch in range(num_epochs):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            y_predicted = model(x_batch)
            loss = criterion(y_predicted, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, loss = {loss.item():.4f}')
            
    y_predicted = model(x)
    mse = criterion(y_predicted, y)
    print('Mean squared error: ', mse.item())

    y_predicted = y_predicted.detach().numpy()

    print('Predicted values:')
    for i in range(len(y_predicted)):
        print(f'Sample {i+1}: {round(y_predicted[i][0]):.2f}, Target = {y[i][0]:.2f}')
    
    plt.scatter(y, y_predicted, alpha=0.5, label="Data Points")
    plt.xlabel("Target Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Target Values")
    plt.legend()
    plt.show()
