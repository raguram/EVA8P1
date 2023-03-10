import torch
from tqdm import tqdm
from cnnlib import metrics_utils, utils


class ModelBuilder:
    """
    Builds the model with for the given parameters. Use fit method with appropriate number of epoch after initializing the object.
    """

    def __init__(self, model, data, lossFn, optimizer, scheduler=None, test_frequency=5):
        self.model = model
        self.lossFn = lossFn
        self.optimizer = optimizer
        self.data = data
        self.scheduler = optimizer if scheduler is None else scheduler
        self.test_frequency = test_frequency
        self.trainer = ModelTrainer()
        self.tester = ModelTester()

    def fit(self, epoch, device=utils.getDevice()):
        train_accs = []
        train_losses = []
        test_accs = []
        test_losses = []
        learning_rates = []
        for e in range(0, epoch):
            print(f'\n\nEpoch: {e + 1}')

            learning_rate = self.optimizer.param_groups[0]['lr']
            learning_rates.append(learning_rate)
            train_result = self.trainer.train_one_epoch(self.model, self.data.train, self.optimizer, device=device,
                                                        lossFn=self.lossFn, scheduler=self.scheduler)
            trainAcc = metrics_utils.compute_accuracy(train_result.predictions, train_result.targets)
            train_accs.append(trainAcc)
            train_losses.append(train_result.loss.to('cpu').detach().numpy())

            print(f'Train Accuracy: {trainAcc}%, Train Loss: {train_result.loss}, Learning Rate: {learning_rate}')

            if e % self.test_frequency == 0 or e + 1 == epoch: 
                test_result = self.tester.test(self.model, self.data.test, lossFn=self.lossFn, device=device)
                testAcc = metrics_utils.compute_accuracy(test_result.predictions, test_result.targets)
                test_accs.append(testAcc)
                test_losses.append(test_result.loss.to('cpu').detach().numpy())
                print(f'Test Accuracy: {testAcc}%, Test Loss: {test_result.loss}')

        return ModelBuildResult(train_accs, train_losses, test_accs, test_losses, learning_rates)


class ModelTrainer:

    def __train_one_batch(self, model, data, target, optimizer, lossFn):
        optimizer.zero_grad()
        output = model(data)
        loss = lossFn(output, target)
        loss.backward()
        optimizer.step()
        return (loss, output.argmax(dim=1))

    def train_one_epoch(self, model, train_loader, optimizer, device, lossFn, scheduler):
        model.train()
        pbar = tqdm(train_loader, ncols=100)
        wholePred = []
        wholeData = []
        wholeTarget = []
        totalLoss = 0
        for idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            (loss, prediction) = self.__train_one_batch(model, data, target, optimizer, lossFn)
            wholePred.append(prediction)
            wholeData.append(data)
            wholeTarget.append(target)
            totalLoss += loss
            scheduler.step()

        return PredictionResult(torch.cat(wholeData), torch.cat(wholePred), torch.cat(wholeTarget),
                                totalLoss / len(train_loader.dataset))


class ModelTester:

    def __test_one_batch(self, model, data, target, lossFn):
        output = model(data)
        loss = lossFn(output, target)
        return (loss, output.argmax(dim=1))

    def test(self, model, loader, lossFn, device=utils.getDevice()):
        model.eval()
        pbar = tqdm(loader, ncols=100)
        wholePred = []
        wholeData = []
        wholeTarget = []
        totalLoss = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                (loss, prediction) = self.__test_one_batch(model, data, target, lossFn)
                totalLoss += loss
                wholePred.append(prediction)
                wholeData.append(data)
                wholeTarget.append(target)

        return PredictionResult(torch.cat(wholeData), torch.cat(wholePred), torch.cat(wholeTarget),
                                totalLoss / len(loader.dataset))


class PredictionResult:

    def __init__(self, data, predictions, targets, loss):
        self.data = data
        self.predictions = predictions
        self.targets = targets
        self.loss = loss


class ModelBuildResult:

    def __init__(self, trainAccuracies, trainLosses, testAccuracies, testLosses, learningRates=None):
        self.trainAccuracies = trainAccuracies
        self.trainLosses = trainLosses
        self.testAccuracies = testAccuracies
        self.testLosses = testLosses
        self.learningRates = learningRates