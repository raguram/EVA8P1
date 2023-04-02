import torch.optim.lr_scheduler as lr_scheduler
import torch

class CustomScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, lr_schedule, train_loader_size):
        self.lr_schedule = lr_schedule
        self.train_loader_size = train_loader_size
        self.count_steps = 0
        super(CustomScheduler, self).__init__(optimizer)

    def get_lr(self, epoch):
        return self.lr_schedule(epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = (self.count_steps + 1) / self.train_loader_size

        self.count_steps += 1
            
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group.update(lr=lr)

def __main__():
    
    from models import ultimus
    import numpy as np

    model = ultimus.Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    epochs = 10
    lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], 
                                  [0, 0.01, 0.01/20.0, 0])[0]
    
    scheduler = CustomScheduler(optimizer, lr_schedule, 5)

    for e in range(50):
        scheduler.step()
        learning_rate = optimizer.param_groups[0]['lr']
        print(learning_rate)

if __name__ == "__main__":
    __main__()

