import torch


class FullyBayesian():

    def __init__(self, size_of_marginal, model, testloader, enable_cuda):
        self.enable_cuda = enable_cuda
        if self.enable_cuda:
            self.counter = torch.ones(1).cuda()
            self.marginal = torch.zeros(size_of_marginal).cuda()
        else:
            self.counter = torch.ones(1)
            self.marginal = torch.zeros(size_of_marginal)
        self.testloader = testloader
        self.model = model
        self.softmax = torch.nn.Softmax(dim=-1)

    def update(self, predict):
        self.marginal = self.marginal + 1 / self.counter * (predict - self.marginal)
        self.counter.add_(1)

    def get_marginal(self):
        return self.marginal

    def evaluation(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.testloader):
                if self.enable_cuda:
                    x, y = x.cuda(), y.cuda()
                likelihood = self.softmax(self.model(x))
                self.update(likelihood.data)
                marginal = self.get_marginal()
                _, yhat = torch.max(marginal, 1)
                total += y.size(0)
                correct += (yhat == y.data).sum()
        return float(correct) / float(total) * 100


class PointEstimate():

    def __init__(self, model, testloader, enable_cuda):
        self.model = model
        self.testloader = testloader
        self.enable_cuda = enable_cuda

    def evaluation(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.testloader:
                x, y = data
                if self.enable_cuda:
                    x, y = x.cuda(), y.cuda()
                outputs = self.model(x)
                _, yhat = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (yhat == y).sum().item()
        return float(correct) / float(total) * 100
