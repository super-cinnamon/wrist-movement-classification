import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, activation='relu',
                 optimizer='adam', batch_size=32, regularization=0.0001,
                 epochs=100, lr=0.01, loss='hinge',
                 report=True):
        super(MultiLayerPerceptron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # Default to 32 if not specified
        self.output_size = output_size
        self.activation = activation
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.regularization = regularization
        self.epochs = epochs
        self.lr = lr
        self.loss_name = loss
        self.report = report

        # Define layers
        self.fc1 = nn.Linear(input_size, self.hidden_size)  # Input layer
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)  # Hidden layer
        self.fc3 = nn.Linear(self.hidden_size, output_size)  # Output layer

        # Activation functions
        self.activation_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'sigmoid': nn.Sigmoid(),
        }
        self.activation_fn = self.activation_dict.get(self.activation, nn.ReLU())

        # Optimizers
        self.optimizer_dict = {
            'adam': optim.Adam,
            'rmsprop': optim.RMSprop,
            'sgd': optim.SGD
        }
        self.optimizer_fn = self.optimizer_dict.get(self.optimizer_name, optim.Adam)

        # Loss functions
        self.loss_dict = {
            'mse': nn.MSELoss(),
            'hinge': nn.HingeEmbeddingLoss(),
            'log': nn.BCEWithLogitsLoss(),
        }
        self.loss_fn = self.loss_dict.get(self.loss_name, nn.BCEWithLogitsLoss())

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation_fn(out)
        out = self.fc2(out)
        out = self.activation_fn(out)
        out = self.fc3(out)
        out = self.activation_dict['sigmoid'](out)
        return out

    def train_model(self, train_loader, val_loader=None):
        optimizer = self.optimizer_fn(self.parameters(), lr=self.lr, weight_decay=self.regularization)
        criterion = self.loss_fn

        best_f1 = 0
        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            if val_loader:
                f1 = self.evaluate(val_loader)
                if f1 > best_f1:
                    best_f1 = f1
                if self.report:
                    print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, F1: {f1:.2f}%')
            else:
                if self.report:
                    print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}')

        return best_f1

    def evaluate(self, dataloader):
        self.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                predicted = (outputs > 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='binary') * 100
        return f1

    def predict(self, data):
        """
        This function takes a list of floats as input and returns a single float between 0 and 1.
        """
        if len(data) != self.input_size:
            raise ValueError(f"Input data must be a list of {self.input_size} floats.")

        # Convert data to a tensor
        data_tensor = torch.tensor(data, dtype=torch.float).unsqueeze(0)  # Add batch dimension

        # Forward pass through the network
        self.eval()
        with torch.no_grad():
            output = self.forward(data_tensor)

        return output.item()


def data_loader(X, y, batch_size):
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def grid_search(
        X_train, y_train, X_val, y_val,
        hyperparameter_combinations,
        input_size,
        MLP_PATH
):

    results = []

    # Iterate through each combination
    for idx, (hidden_size, activation, optimizer_name, batch_size,
              regularization, epochs, lr, loss_name) in enumerate(hyperparameter_combinations, 1):    # noqa

        print(f"\nRunning combination {idx}/{len(hyperparameter_combinations)}:")
        print(f"Hidden Size: {hidden_size}, Activation: {activation}, Optimizer: {optimizer_name}, "  # noqa
            f"Batch Size: {batch_size}, Regularization: {regularization}, Epochs: {epochs}, "         # noqa
            f"Learning Rate: {lr}, Loss: {loss_name}")

        # Create DataLoaders with the current batch size
        train_loader = data_loader(X_train, y_train, batch_size)
        val_loader = data_loader(X_val, y_val, batch_size)

        # Initialize the model with current hyperparameters
        model = MultiLayerPerceptron(
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            optimizer=optimizer_name,
            batch_size=batch_size,
            regularization=regularization,
            epochs=epochs,
            lr=lr,
            loss=loss_name,
            report=False
        )

        # Train the model and get the best F1 score
        try:
            best_f1 = model.train_model(train_loader, val_loader)
        except Exception as e:
            print(f"Error during training: {e}")
            best_f1 = None

        # Record the results
        results.append({
            'hidden_size': hidden_size,
            'activation': activation,
            'optimizer': optimizer_name,
            'batch_size': batch_size,
            'regularization': regularization,
            'epochs': epochs,
            'learning_rate': lr,
            'loss': loss_name,
            'best_f1': best_f1
        })

        # Save intermediate results to prevent data loss in case of interruptions
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(MLP_PATH, 'grid_search_results.csv'), index=False)
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Drop combinations that failed (if any)
    results_df = results_df.dropna(subset=['best_f1'])

    # Find the best hyperparameter set
    best_result = results_df.loc[results_df['best_f1'].idxmax()]

    print("\nBest Hyperparameter Combination:")
    print(best_result)

    # transform into a dict
    best_result = best_result.to_dict()
    return best_result
