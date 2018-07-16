import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, config, model, memory, device):
        batch_size = config.getint('trainer', 'batch_size')
        learning_rate = config.getfloat('trainer', 'learning_rate')
        step_size = config.getint('trainer', 'step_size')
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = DataLoader(memory, batch_size, shuffle=True)
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)

    def optimize_batch(self, num_epochs):
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()

                outputs = self.model(inputs, self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            # logging.info('Loss in epoch {} is {}'.format(epoch, epoch_loss))
            losses.append(epoch_loss / len(self.memory))
        average_epoch_loss = sum(losses) / len(losses)
        return average_epoch_loss
