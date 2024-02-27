import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

class CNN(pl.LightningModule):
    def __init__(self, input_size, NUM_OF_INTERVALS):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(1 * input_size, 1024)
        self.fc1.double()
        self.fc2 = nn.Linear(1024, 512)
        self.fc2.double()
        self.fc3 = nn.Linear(512, 128)
        self.fc3.double()
        self.fc4 = nn.Linear(128, NUM_OF_INTERVALS*2)
        self.fc4.double()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.double()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.L1Loss()(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.L1Loss()(y_pred, y)
        self.log('val_loss', loss)

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def plot_metrics(loss_list, val_loss_list, mse_list, rmse_list, mae_list, r2_list, number_of_epochs):    
    avg_loss_list = loss_list[::number_of_epochs]
    avg_val_loss_list = val_loss_list[::number_of_epochs]
    avg_mse_list = [sum(sublist[i] for sublist in mse_list) / len(mse_list) for i in range(number_of_epochs)]
    avg_rmse_list = [sum(sublist[i] for sublist in rmse_list) / len(rmse_list) for i in range(number_of_epochs)]
    avg_mae_list = [sum(sublist[i] for sublist in mae_list) / len(mae_list) for i in range(number_of_epochs)]
    avg_r2_list = [sum(sublist[i] for sublist in r2_list) / len(r2_list) for i in range(number_of_epochs)]

    epochs = np.arange(1, number_of_epochs + 1)
    folds = np.arange(1, len(avg_loss_list) + 1)
    
    plt.figure(figsize=(8, 4))
    plt.plot(folds, avg_loss_list, label='Average Train Loss')
    plt.plot(folds, avg_val_loss_list, label='Average Validation Loss')
    plt.xlabel('Folds')
    plt.ylabel('Loss')
    plt.title('Average Training and Validation Loss Over Folds')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, avg_mse_list[:number_of_epochs], label='Average MSE')
    plt.plot(epochs, avg_rmse_list[:number_of_epochs], label='Average RMSE')
    plt.plot(epochs, avg_mae_list[:number_of_epochs], label='Average MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Average Metrics (MSE, RMSE, MAE) Over Training and Testing Period')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, avg_r2_list[:number_of_epochs], label='Average R^2')
    plt.xlabel('Epochs')
    plt.ylabel('R^2')
    plt.title('Average R^2 Over Training and Testing Period')
    plt.legend()
    plt.show()

def kfold_model(X, y, num_folds=5, max_epochs = 31, intervals = 21):
    input_size = X.shape[1]
    num_samples = X.shape[0]
    num_splits = num_folds
    
    NUM_OF_INTERVALS = intervals
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    all_metrics = []
    all_losses = []
    all_val_losses = []
    all_mse_list = []
    all_rmse_list = []
    all_mae_list = []
    all_r2_list = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold_idx + 1}")
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Increase num_workers to 12
        val_loader = DataLoader(val_dataset, batch_size=16)  # Increase num_workers to 12

        model = CNN(input_size, NUM_OF_INTERVALS)
        logger = TensorBoardLogger("logs/", name=f"fold_{fold_idx + 1}")
        trainer = pl.Trainer(max_epochs=max_epochs, logger=logger)
        trainer.fit(model, train_loader, val_loader)

        loss_list = [trainer.callback_metrics['train_loss'].item() for epoch in range(1, max_epochs)]
        val_loss_list = [trainer.callback_metrics['val_loss'].item() for epoch in range(1, max_epochs)]
        all_losses.extend(loss_list)
        all_val_losses.extend(val_loss_list)
        
        # Initialize lists to store metric values for each epoch
        mse_list = [0] * (max_epochs - 1)
        rmse_list = [0] * (max_epochs - 1)
        mae_list = [0] * (max_epochs - 1)
        r2_list = [0] * (max_epochs - 1)

        model.eval()
        with torch.no_grad():
            val_true = []
            val_pred = []
            for batch in val_loader:
                x_val, y_val = batch
                y_val_pred = model(x_val)
                val_true.append(y_val.numpy())
                val_pred.append(y_val_pred.numpy())

            val_true = np.concatenate(val_true)
            val_pred = np.concatenate(val_pred)
            metrics = evaluate(val_true, val_pred)
            all_metrics.append(metrics)

            # Calculate and store metric values for each epoch
            for epoch in range(max_epochs - 1):
                val_true_epoch = val_true[epoch::(max_epochs - 1)]
                val_pred_epoch = val_pred[epoch::(max_epochs - 1)]

                if len(val_true_epoch) >= 2:  # Check if there are enough samples for R^2 calculation
                    mse_list[epoch] = mean_squared_error(val_true_epoch, val_pred_epoch)
                    rmse_list[epoch] = np.sqrt(mse_list[epoch])
                    mae_list[epoch] = mean_absolute_error(val_true_epoch, val_pred_epoch)
                    r2_list[epoch] = r2_score(val_true_epoch, val_pred_epoch)
                else:
                    mse_list[epoch] = 0  # You can set it to any default value or handle it differently
                    rmse_list[epoch] = 0
                    mae_list[epoch] = 0
                    r2_list[epoch] = 0

        all_mse_list.append(mse_list)
        all_rmse_list.append(rmse_list)
        all_mae_list.append(mae_list)
        all_r2_list.append(r2_list)

        model.train()

    for fold_idx, metrics in enumerate(all_metrics):
        mse, rmse, mae, r2 = metrics
        print(f"Fold {fold_idx + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    avg_mse = np.mean([metrics[0] for metrics in all_metrics])  # Calculate average MSE
    avg_rmse = np.mean([metrics[1] for metrics in all_metrics])  # Calculate average RMSE
    avg_mae = np.mean([metrics[2] for metrics in all_metrics])  # Calculate average MAE
    avg_r2 = np.mean([metrics[3] for metrics in all_metrics])   # Calculate average R2

    print(f"Average MSE across all folds: {avg_mse:.4f}")
    print(f"Average RMSE across all folds: {avg_rmse:.4f}")
    print(f"Average MAE across all folds: {avg_mae:.4f}")
    print(f"Average R^2 across all folds: {avg_r2:.4f}")
    
    number_of_epochs = max_epochs - 1    
    # plot_metrics(all_losses, all_val_losses, all_mse_list, all_rmse_list, all_mae_list, all_r2_list, number_of_epochs)
    return model