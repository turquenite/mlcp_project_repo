from models import SimpleModel, DeepModel
from train import train
from dataset import MLPC_Dataset

dataset = MLPC_Dataset()
train_loader, test_loader = dataset.get_data_loaders(batch_size=32, split=(0.75, 0.25))

model = SimpleModel()

train(network=model, test_loader=test_loader, train_loader=train_loader,num_epochs=100, plot_loss=True, save_model=True, plot_accuracy=True)