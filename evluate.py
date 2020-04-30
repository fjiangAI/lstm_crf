class Acc:
    def __init__(self):
        self.acc = 0
        self.correct = 0
        self.count = 0

    def update(self, pred, golden):
        pred=pred.detach().cpu().numpy()
        golden=golden.detach().cpu().numpy()
        train_correct = (pred == golden).sum()
        self.correct += train_correct
        self.count += len(pred)
        self.acc = self.correct / float(self.count)
