from abc import ABCMeta, abstractmethod
from typing import Optional, Union

from fiqa.navigation.basics_and_dummies import NavigatorBase
from fiqa.task_handlers.interactor import InteractorBase


class InfoRetrieverBase(metaclass=ABCMeta):
    """Sets an interface for a class that retrieves information 
    for a navigator or an interactor from the execution process.
    
    Such information can include the subtask completion success, 
    the interaction mask used and so on.
    """

    def __init__(
        self, predictor: Union[NavigatorBase, InteractorBase]
    ) -> None:
        self.predictor = predictor

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update_predictor_values(
        self, subtask_success: Optional[bool] = None
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_info(self, *args, **kwargs) -> None:
        """The function is used only to save information."""
        raise NotImplementedError()


class DummyInfoRetriever(InfoRetrieverBase):
    """This retriever doesn't retrieve anything and is used for navigators and 
    interactors that don't need any info besides the `rgb`, `subtask` and 
    `retry_nav`."""

    def __init__(
        self, predictor: Optional[Union[NavigatorBase, InteractorBase]] = None
    ) -> None:
        super().__init__(predictor)

    def reset(self) -> None:
        pass

    def update_predictor_values(
        self, subtask_success: Optional[bool]
    ) -> None:
        pass

    def save_info(self, *args, **kwargs) -> None:
        pass
