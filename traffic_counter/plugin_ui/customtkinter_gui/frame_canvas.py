from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customtkinter import NW
from PIL import Image, ImageTk

from traffic_counter.adapter_ui.abstract_canvas import AbstractCanvas
from traffic_counter.adapter_ui.abstract_frame_canvas import AbstractFrameCanvas
from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.application.logger import logger
from traffic_counter.domain.track import TrackImage, PilImage
from traffic_counter.plugin_ui.customtkinter_gui.canvas_observer import (
    CanvasObserver,
    EventHandler,
)
from traffic_counter.plugin_ui.customtkinter_gui.constants import (
    DELETE_KEYS,
    ENTER_CANVAS,
    ESCAPE_KEY,
    LEAVE_CANVAS,
    LEFT_BUTTON_DOWN,
    LEFT_BUTTON_UP,
    LEFT_KEY,
    MOTION,
    MOTION_WHILE_LEFT_BUTTON_DOWN,
    PADX,
    PLUS_KEYS,
    RETURN_KEY,
    RIGHT_BUTTON_UP,
    RIGHT_KEY,
    STICKY,
    tk_events,
)
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    EmbeddedCTkFrame,
)
from traffic_counter.plugin_ui.customtkinter_gui.helpers import get_widget_position


@dataclass
class DisplayableImage:
    _image: TrackImage

    def width(self) -> int:
        return self._image.width()

    def height(self) -> int:
        return self._image.height()

    def create_photo(self) -> ImageTk.PhotoImage:
        return ImageTk.PhotoImage(image=self._image.as_image())


class FrameCanvas(AbstractFrameCanvas, EmbeddedCTkFrame):
    def __init__(self, viewmodel: ViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._current_image = None
        self._viewmodel = viewmodel
        self._get_widgets()
        self._place_widgets()
        self.introduce_to_viewmodel()

    def introduce_to_viewmodel(self) -> None:
        self._viewmodel.set_frame_canvas(self)

    def on_window_resize(self, event):
        diff = abs(self.canvas_background._current_image.width() - self.winfo_width())
        if self._current_image and diff > 20:
            factor = self.winfo_width() / self._current_image.width()
            self._viewmodel.update_sections_canvas_coordinates(factor)
            self.add_image(DisplayableImage(self._current_image), layer="background")

    def _get_widgets(self) -> None:
        self.canvas_background = CanvasBackground(
            master=self, viewmodel=self._viewmodel
        )
        self.canvas_background.bind("<Configure>", self.on_window_resize)

    def _place_widgets(self) -> None:
        self.canvas_background.pack(expand=True)

    def update_background(self, image: TrackImage) -> None:
        self._current_image = image
        self.add_image(DisplayableImage(image), layer="background")
        factor = self.winfo_width() / self._current_image.width()
        self._viewmodel.update_sections_canvas_coordinates(factor)

    def resize_image(self, image: DisplayableImage):
        image_width = self.winfo_width()
        factor = image_width / image.width()
        image = image._image.as_image().resize(
            ((int)(image.width() * factor), (int)(image.height() * factor))
        )
        return DisplayableImage(PilImage(image))

    def add_image(self, image: DisplayableImage, layer: str) -> None:
        image = self.resize_image(image)
        self.canvas_background.add_image(image, layer)

    def clear_image(self) -> None:
        self.canvas_background.clear_image()

    def set_current_image(self, image: TrackImage):
        self._current_image = image


class CanvasBackground(AbstractCanvas):
    def __init__(self, viewmodel: ViewModel, **kwargs: Any):
        super().__init__(**kwargs)
        self._viewmodel = viewmodel
        self.logo_img_path = r"traffic_counter/assets/logo.png"
        self.event_handler = CanvasEventHandler(canvas=self)
        self.introduce_to_viewmodel()

        self._current_image: ImageTk.PhotoImage
        self._current_id: Any = None
        self.add_preview_image()
        self.master.set_current_image(PilImage(Image.open(self.logo_img_path)))

    def add_preview_image(self) -> None:
        if Path(self.logo_img_path).exists():
            preview_image = Image.open(self.logo_img_path)
            self._current_image = ImageTk.PhotoImage(preview_image)
            self._draw()

    def add_image(self, image: DisplayableImage, layer: str) -> None:
        if self._current_id:
            self.delete(self._current_id)
        self._current_image = image.create_photo()
        self._draw()

    def _draw(self) -> None:
        self._current_id = self.create_image(0, 0, image=self._current_image, anchor=NW)
        self.config(
            width=self._current_image.width(), height=self._current_image.height()
        )
        self.config(highlightthickness=0)
        self._viewmodel.refresh_items_on_canvas()

    def introduce_to_viewmodel(self) -> None:
        self._viewmodel.set_canvas(self)

    def get_position(self, offset: tuple[float, float] = (0.5, 0.5)) -> tuple[int, int]:
        x, y = get_widget_position(self, offset=offset)
        return x, y

    def clear_image(self) -> None:
        if self._current_id:
            self.delete(self._current_id)
            self._viewmodel.refresh_items_on_canvas()


class CanvasEventHandler(EventHandler):
    def __init__(self, canvas: CanvasBackground):
        self._canvas = canvas
        self._observers: list[CanvasObserver] = []
        self._bind_events()

    def _bind_events(self) -> None:
        self._canvas.bind(tk_events.MOUSE_ENTERS_WIDGET, self._on_mouse_enters_canvas)
        self._canvas.bind(tk_events.MOUSE_LEAVES_WIDGET, self._on_mouse_leaves_canvas)
        self._canvas.bind(tk_events.LEFT_BUTTON_DOWN, self._on_left_button_down)
        self._canvas.bind(tk_events.LEFT_BUTTON_UP, self._on_left_button_up)
        self._canvas.bind(tk_events.RIGHT_BUTTON_UP, self._on_right_button_up)
        self._canvas.bind(tk_events.MOUSE_MOTION, self._on_mouse_motion)
        self._canvas.bind(
            tk_events.MOUSE_MOTION_WHILE_LEFT_BUTTON_DOWN,
            self._on_motion_while_left_button_down,
        )
        self._canvas.bind(tk_events.PLUS_KEY, self._on_plus)
        self._canvas.bind(tk_events.KEYPAD_PLUS_KEY, self._on_plus)
        self._canvas.bind(tk_events.LEFT_ARROW_KEY, self._on_left)
        self._canvas.bind(tk_events.RIGHT_ARROW_KEY, self._on_right)
        self._canvas.bind(tk_events.RETURN_KEY, self._on_return)
        self._canvas.bind(tk_events.KEYPAD_RETURN_KEY, self._on_return)
        self._canvas.bind(tk_events.DELETE_KEY, self._on_delete)
        self._canvas.bind(tk_events.BACKSPACE_KEY, self._on_delete)
        self._canvas.bind(tk_events.ESCAPE_KEY, self._on_escape)

    def attach_observer(self, observer: CanvasObserver) -> None:
        self._canvas.focus_set()
        self._observers.append(observer)

    def detach_observer(self, observer: CanvasObserver) -> None:
        if self._canvas.focus_get() == self._canvas:
            logger().debug("Set focus to canvases masters master")
            self._canvas.master.master.focus_set()
        self._observers.remove(observer)

    def _notify_observers(
        self, event: Any, event_type: str, key: str | None = None
    ) -> None:
        coordinates = self._get_mouse_coordinates(event)
        for observer in self._observers:
            observer.update(coordinates, event_type, key)

    def _on_left_button_down(self, event: Any) -> None:
        self._notify_observers(event, LEFT_BUTTON_DOWN)

    def _on_left_button_up(self, event: Any) -> None:
        self._notify_observers(event, LEFT_BUTTON_UP)

    def _on_right_button_up(self, event: Any) -> None:
        self._notify_observers(event, RIGHT_BUTTON_UP)

    def _on_mouse_motion(self, event: Any) -> None:
        self._notify_observers(event, MOTION)

    def _on_motion_while_left_button_down(self, event: Any) -> None:
        self._notify_observers(event, MOTION_WHILE_LEFT_BUTTON_DOWN)

    def _on_mouse_leaves_canvas(self, event: Any) -> None:
        self._notify_observers(event, LEAVE_CANVAS)

    def _on_mouse_enters_canvas(self, event: Any) -> None:
        self._notify_observers(event, ENTER_CANVAS)

    def _on_plus(self, event: Any) -> None:
        self._notify_observers(event, PLUS_KEYS)

    def _on_left(self, event: Any) -> None:
        self._notify_observers(event, LEFT_KEY)

    def _on_right(self, event: Any) -> None:
        self._notify_observers(event, RIGHT_KEY)

    def _on_return(self, event: Any) -> None:
        self._notify_observers(event, RETURN_KEY)

    def _on_delete(self, event: Any) -> None:
        self._notify_observers(event, DELETE_KEYS)

    def _on_escape(self, event: Any) -> None:
        self._notify_observers(event, ESCAPE_KEY)

    def _get_mouse_coordinates(self, event: Any) -> tuple[int, int]:
        """Returns coordinates of event on canvas taking into account the horizontal and
        vertical scroll factors ("xscroll" and "yscroll") in case the canvas is zoomed
        and scrolled.

        Args:
            event (Any): Event on canvas

        Returns:
            tuple[int, int]: Coordinates of the event
        """
        x = int(self._canvas.canvasx(event.x))
        y = int(self._canvas.canvasy(event.y))
        return x, y
