from pathlib import Path
from typing import Any

from PIL import Image

from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.domain.video import Video
from customtkinter import CTkLabel, CTkImage
from traffic_counter.plugin_ui.customtkinter_gui.constants import PADY, PADX, STICKY
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    CustomCTkTabview,
    EmbeddedCTkFrame,
)


class FrameFile(EmbeddedCTkFrame):
    status_img_paths = {
        True: Path(r"traffic_counter/assets/is_analyzed.png"),
        False: Path(r"traffic_counter/assets/is_not_analyzed.png"),
    }

    def __init__(
        self,
        parent,
        viewmodel: ViewModel,
        file_path: str,
        is_analyzed: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.parent = parent
        self._viewmodel = viewmodel
        self.file_path = file_path
        self.filename = file_path.name
        self.is_analyzed = is_analyzed
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self._label_filename = CTkLabel(master=self, text=self.filename)
        self._label_filename.bind("<Button-1>", self.select_file)
        self._label_status = CTkLabel(master=self, text="")
        self.set_status(self.is_analyzed)

    def select_file(self, event=None):
        self._viewmodel.set_selected_videos([self.file_path])
        self.configure(fg_color="#3076FF")

    def set_status(self, is_analyzed):
        status_img = CTkImage(
            light_image=Image.open(self.status_img_paths[is_analyzed]),
            size=(20, 20),
        )
        self._label_status.configure(image=status_img)

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
        self._viewmodel.set_treeview_files(self)

    def _get_widgets(self) -> None:
        self.add(self._title)

    def _place_widgets(self) -> None:
        for i, file in enumerate(self.files):
            file.grid(row=i + 1, column=0, pady=PADY, padx=PADX)

    def update_items(self) -> None:
        self.files = [
            self.__to_resource(video) for video in self._viewmodel.get_all_videos()
        ]
        for file in self.files:
            for track in self._viewmodel.get_all_track_files():
                if track.name.rsplit(".")[0] == file.filename.rsplit(".")[0]:
                    file.set_status(True)

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
            is_analyzed=False,
        )
