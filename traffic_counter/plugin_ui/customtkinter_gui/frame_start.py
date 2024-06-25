from typing import Any

from customtkinter import CTkButton

from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.plugin_ui.customtkinter_gui.constants import PADY, STICKY
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    CustomCTkTabview,
    EmbeddedCTkFrame,
)


class TabviewStart(CustomCTkTabview):
    def __init__(
        self,
        viewmodel: ViewModel,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._viewmodel = viewmodel
        self._title: str = "Start"
        self._get_widgets()
        self._place_widgets()
        self.disable_segmented_button()

    def _get_widgets(self) -> None:
        self.add(self._title)
        self.frame_start = FrameStart(
            master=self.tab(self._title), viewmodel=self._viewmodel
        )

    def _place_widgets(self) -> None:
        self.frame_start.pack(expand=True)
        self.set(self._title)


class FrameStart(EmbeddedCTkFrame):
    def __init__(self, viewmodel: ViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._viewmodel = viewmodel
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self._button_frame = EmbeddedCTkFrame(master=self)

        self.button_load_video = CTkButton(
            master=self._button_frame,
            text="Load video",
            height=35,
            width=180,
            command=self._viewmodel.add_video,
            font=("Helvetica", 14),
        )
        self.button_load_track = CTkButton(
            master=self._button_frame,
            text="Load analyzed track file",
            height=35,
            width=180,
            command=self._viewmodel.load_tracks,
            font=("Helvetica", 14),
        )

    def _place_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._button_frame.grid(
            row=0, column=0, columnspan=2, padx=0, pady=0, sticky=STICKY
        )
        self.button_load_video.pack(pady=PADY * 2)
        self.button_load_track.pack(pady=PADY * 2)
