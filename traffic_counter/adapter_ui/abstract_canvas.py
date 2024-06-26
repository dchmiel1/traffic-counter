from abc import abstractmethod

from customtkinter import CTkCanvas

from traffic_counter.adapter_ui.helpers import WidgetPositionProvider

# from OTAnalytics.plugin_ui.canvas_observer import EventHandler


class AbstractCanvas(CTkCanvas, WidgetPositionProvider):
    # TODO: Properly define abstract property here and in derived class(es)
    # @property
    # @abstractmethod
    # def event_handler(self) -> EventHandler:
    #     pass

    # TODO: Define whole interface (all properties and methods) required by viewmodel

    @abstractmethod
    def introduce_to_viewmodel(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_preview_image(self) -> None:
        raise NotImplementedError
