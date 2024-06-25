from typing import Any

from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.domain.video import Video
from customtkinter import CTkLabel
from traffic_counter.plugin_ui.customtkinter_gui.constants import PADY, PADX, STICKY
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    CustomCTkTabview,
    EmbeddedCTkFrame,
)


class FrameFile(EmbeddedCTkFrame):
    def __init__(
        self, parent, viewmodel: ViewModel, file_path: str, status: bool, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.parent = parent
        self._viewmodel = viewmodel
        self.file_path = file_path
        self.filename = file_path.name
        self.status = status
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self._label_filename = CTkLabel(master=self, text=self.filename)
        self._label_status = CTkLabel(master=self, text=self.status)
        self._label_filename.bind("<Button-1>", self.select_file)

    def select_file(self, event=None):
        self._viewmodel.set_selected_videos([self.file_path])
        self.configure(fg_color="#3076FF")

    def _place_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._label_filename.grid(row=0, column=0, padx=0, pady=0, sticky=STICKY)
        self._label_status.grid(row=0, column=1, padx=0, pady=0, sticky=STICKY)


class TabviewFiles(CustomCTkTabview):
    def __init__(
        self,
        viewmodel: ViewModel,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._viewmodel = viewmodel
        self._title: str = "Files"
        self.files = []
        self._get_widgets()
        self._place_widgets()
        self.disable_segmented_button()
        self._introduce_to_viewmodel()

    def _introduce_to_viewmodel(self) -> None:
        self._viewmodel.set_treeview_videos(self)

    def _get_widgets(self) -> None:
        self.add(self._title)

    def _place_widgets(self) -> None:
        for i, file in enumerate(self.files):
            file.grid(row=i + 1, column=0, pady=PADY, padx=PADX)

    def update_items(self) -> None:
        self.files = [
            self.__to_resource(video) for video in self._viewmodel.get_all_videos()
        ]
        self._place_widgets()

    def update_selected_items(self, item_ids: list[str]):
        self.unselect_all()
        for file in self.files:
            if str(file.file_path) == item_ids[0]:
                file.select_file()

    def unselect_all(self):
        for file in self.files:
            file.configure(fg_color="transparent")

    def __to_resource(self, video: Video) -> FrameFile:
        return FrameFile(
            parent=self,
            master=self.tab(self._title),
            viewmodel=self._viewmodel,
            file_path=video.get_path(),
            status="status",
        )
