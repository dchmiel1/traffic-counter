from abc import ABC, abstractmethod

from traffic_counter.adapter_ui.helpers import WidgetPositionProvider


class AbstractMainWindow(WidgetPositionProvider, ABC):
    @abstractmethod
    def introduce_to_viewmodel(self) -> None:
        raise NotImplementedError
