import abc

class binary_model(metaclass=abc.ABCMeta):
    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass

    @abc.abstractmethod
    def score(self, y_test, y_preds, pos_label=1):
        pass