import abc


class Policy(object):
    def __init__(self):
        ...

    @abc.abstractmethod
    def configure(self, config):
        ...

    @abc.abstractmethod
    def predict(self, **kwargs):
        """
        Policy takes state as input and output an action

        """
        ...