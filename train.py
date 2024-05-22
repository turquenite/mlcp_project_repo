import torch
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping

def train(network: torch.nn.Module, test_loader: DataLoader, train_loader: DataLoader, num_epochs: int = 10, lr:float = 0.1, optimizer_type = torch.optim.Adam, loss_function = torch.nn.CrossEntropyLoss(), plot_loss: bool = False, save_model: bool = False, plot_accuracy: bool = False):
    optimizer = optimizer_type(network.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 10, gamma=0.1)
    early_stopping = EarlyStopping(patience=5)

    epoch_train_losses = list()
    epoch_test_losses = list()
    epoch_train_acc = list()
    epoch_test_acc = list()
    

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network.to(device)

    for _ in tqdm(range(num_epochs), total=num_epochs, desc=f"Training in {num_epochs} Epochs"):
        network.train()

        minibatch_training_loss = list()
        minibatch_test_loss = list()
        minibatch_test_accuraccies = list()
        minibatch_train_accuraccies = list()
        
        for batch in train_loader:
            network_input, insight_values, targets = batch
            network_input = network_input.to(device)
            targets = targets.to(device)
            
            # Get Model Output (forward pass)
            output = network(network_input)

            # Compute Loss
            loss = loss_function(output, targets)

            # Compute gradients (backward pass)
            loss.backward()  

            # Perform gradient descent update step
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            minibatch_training_loss.append(loss.item())
            minibatch_train_accuraccies.append(np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(targets.cpu().detach().numpy(), axis=1)))

        # Update the learning rate at the end of each epoch
        scheduler.step()
        epoch_train_acc.append(np.mean(minibatch_train_accuraccies))
        epoch_train_losses.append(np.mean(minibatch_training_loss))

        print(f"Train metrics:\nLoss: {epoch_train_losses[-1]}\nAccuracy: {epoch_train_acc[-1]}\n")

        network.eval()

        for batch in test_loader:
            network_input, insight_values, targets = batch

            network_input = network_input.to(device)
            targets = targets.to(device)
            output = network(network_input)
            minibatch_test_accuraccies.append(np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(targets.cpu().detach().numpy(), axis=1)))
            loss = loss_function(output, targets)
            minibatch_test_loss.append(loss.item())

        epoch_test_losses.append(np.mean(minibatch_test_loss))
        epoch_test_acc.append(np.mean(minibatch_test_accuraccies))

        print(f"Test metrics:\nLoss: {epoch_test_losses[-1]}\nAccuracy: {epoch_test_acc[-1]}\n")


        # Early Stopping
        early_stopping(np.mean(epoch_test_losses))

        if early_stopping.early_stop:    
            break

    if plot_loss:
        plot_losses(epoch_train_losses, epoch_test_losses)

    if plot_accuracy:
        plot_accuracies(epoch_train_acc, epoch_test_acc)

    if save_model:
        scripted_model = torch.jit.script(network)

        cur_time = datetime.now().strftime("%d-%m-%Y_%H;%M")
        folder_path = fr"{os.getcwd()}\models\{cur_time}"

        os.mkdir(folder_path)
        scripted_model.save(folder_path + r"\model.pt")

        with open(folder_path + r"\metadata.txt", "w") as f:
            f.write(f"DateTime: {cur_time}\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Optimizer: {optimizer_type}\n")
            f.write(f"Loss Function: {loss_function}\n")
            f.write(f"Train Losses: {epoch_train_losses}\n")
            f.write(f"Test Losses: {epoch_test_losses}\n")
            f.write(f"Train Accuracy: {np.mean(epoch_train_acc)}\n")
            f.write(f"Test Accuracy: {np.mean(epoch_test_acc)}\n\n")
            f.write(f"Architecture: \n{network}")

    return (epoch_train_losses, epoch_test_losses)

def plot_losses(train_losses: list, test_losses: list):
    plt.title("Model-Loss")
    plt.plot(range(len(train_losses)), train_losses, "orange", range(len(test_losses)), test_losses, "dodgerblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training Loss", "Test Loss"])
    plt.show()

def plot_accuracies(train_acc, test_acc: list):
    plt.title("Model Accuracies")
    plt.plot(range(len(train_acc)), train_acc, "orange", range(len(test_acc)), test_acc, "dodgerblue")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training Accuracy", "Test Accuracy"])
    plt.show()