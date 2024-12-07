class SuperVision:
    def __init__(self, model, loss, optimizer, device):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item()}')
        print(f'Epoch: {epoch} Train Loss: {train_loss/len(train_loader)} Train Acc: {100.*correct/total}')

    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print(f'Epoch: {epoch} [{batch_idx}/{len(test_loader)}] Loss: {loss.item()}')
            print(f'Epoch: {epoch} Test Loss: {test_loss/len(test_loader)} Test Acc: {100.*correct/total}')