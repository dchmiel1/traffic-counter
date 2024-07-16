from pathlib import Path
from typing import Any

from PIL import Image

from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.domain.video import Video
from customtkinter import CTkLabel, CTkImage, CTkFont
from traffic_counter.plugin_ui.customtkinter_gui.constants import PADY, PADX, STICKY
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    CustomCTkTabview,
    EmbeddedCTkFrame,
)


class FrameFile(EmbeddedCTkFrame):
    status_img_paths = {
        True: Path(r"traffic_counter/assets/is_processed.png"),
        False: Path(r"traffic_counter/assets/is_not_processed.png"),
    }

    def __init__(
        self,
        parent,
        viewmodel: ViewModel,
        file_path: str,
        is_processed: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.parent = parent
        self._viewmodel = viewmodel
        self.file_path = file_path
        self.filename = file_path.name
        self.is_processed = is_processed
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self._label_filename = CTkLabel(master=self, text=self.filename, width=250)
        self._label_filename.bind("<Button-1>", self.select)
        self._label_status = CTkLabel(master=self, text="")
        self._label_status.bind("<Button-1>", self.select)
        self.set_status(self.is_processed)

    def select(self, event=None):
        self._viewmodel.set_selected_videos([self.file_path])
        self.configure(fg_color="#3076FF")
        self._label_filename.configure(font=CTkFont(weight="bold"))

    def unselect(self):
        self.configure(fg_color="transparent")
        self._label_filename.configure(font=CTkFont())

    def set_status(self, is_processed):
        status_img = CTkImage(
            light_image=Image.open(self.status_img_paths[is_processed]),
            size=(15, 15),
        )
        self._label_status.configure(image=status_img)

    def _place_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        self._label_filename.grid(row=0, column=0, padx=PADX, pady=0, sticky=STICKY)
        self._label_status.grid(row=0, column=1, padx=PADX, pady=0, sticky=STICKY)


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
        self.grid_columnconfigure(0, weight=1)
        for i, file in enumerate(self.files):
            file.grid(row=i + 1, column=0, pady=0, padx=PADX, sticky=STICKY)
            file.bind("<Button-1>", file.select)

    def _update_video_files(self):
        curr_files_paths = [file.file_path for file in self.files]
        for video_file in self._viewmodel.get_all_videos():
            if video_file.get_path() in curr_files_paths:
                continue
            self.files.append(self.__to_resource(video_file))

    def _update_track_files(self):
        # update track files
        for track in self._viewmodel.get_all_track_files():
            for file in self.files:
                if track.name.rsplit(".")[0] == file.filename.rsplit(".")[0]:
                    file.set_status(True)

    def update_items(self) -> None:
        self._update_video_files()
        self._update_track_files()
        self._place_widgets()

    def update_selected_items(self, item_ids: list[str]):
        self.unselect_all()
        if not len(item_ids):
            return
        for file in self.files:
            if str(file.file_path) == item_ids[0]:
                file.select()

    def unselect_all(self):
        for file in self.files:
            file.unselect()

    def __to_resource(self, video: Video) -> FrameFile:
        return FrameFile(
            parent=self,
            master=self.tab(self._title),
            viewmodel=self._viewmodel,
            file_path=video.get_path(),
            is_processed=False,
        )
