from abc import abstractmethod, ABCMeta


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def run(self, fetchResult):
        pass
