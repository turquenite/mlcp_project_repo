from models import SimpleModel, DeepModel, ConvModelShallow
from train import train
from dataset import MLPC_Dataset
import torch

dataset = MLPC_Dataset("mel")
train_loader, test_loader = dataset.get_data_loaders(batch_size=32, split=(0.75, 0.25))

model = ConvModelShallow()

torch.save(model.state_dict(), "ConvShallow_sd")

train(network=model, test_loader=test_loader, train_loader=train_loader,num_epochs=100, plot_loss=True, save_model=True, plot_accuracy=True)

torch.save(model.state_dict(), "ConvShallow_sd")