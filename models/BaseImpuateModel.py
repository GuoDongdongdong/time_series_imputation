import torch
from abc import ABC, abstractmethod


'''
    All model except the STATISTICAL_MODEL_LIST model need to implement the method in BaseImputeModel.
'''
class BaseImputeModel(ABC):
    def forward(self, batch:dict, training:bool) -> torch.Tensor | dict:
        pass

    @abstractmethod
    def evaluate(self, batch:dict, training:bool=True) -> torch.Tensor:
        loss = self.forward(batch, training)
        return loss

    @abstractmethod
    def impute(self, batch:dict, n_sample:int) -> torch.Tensor:
        imputation = self.forward(batch, False)
        return imputation