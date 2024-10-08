import tkinter
from typing import Any, Sequence

from customtkinter import (
    CTk,
    CTkButton,
    CTkFrame,
    CTkFont,
    set_appearance_mode,
    set_default_color_theme,
)

from traffic_counter.adapter_ui.abstract_main_window import AbstractMainWindow
from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.application.exception import gather_exception_messages
from traffic_counter.application.logger import logger
from traffic_counter.application.plotting import Layer
from traffic_counter.plugin_ui.customtkinter_gui.constants import PADX, PADY, STICKY
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    CustomCTkTabview,
    EmbeddedCTkScrollableFrame,
)
from traffic_counter.plugin_ui.customtkinter_gui.frame_analysis import TabviewAnalysis
from traffic_counter.plugin_ui.customtkinter_gui.frame_canvas import FrameCanvas
from traffic_counter.plugin_ui.customtkinter_gui.frame_configuration import (
    TabviewConfiguration,
)
from traffic_counter.plugin_ui.customtkinter_gui.frame_files import FrameFiles
from traffic_counter.plugin_ui.customtkinter_gui.frame_filter import FrameFilter
from traffic_counter.plugin_ui.customtkinter_gui.frame_project import TabviewProject
from traffic_counter.plugin_ui.customtkinter_gui.frame_track_plotting import (
    FrameTrackPlotting,
)
from traffic_counter.plugin_ui.customtkinter_gui.frame_input_files import (
    TabviewFiles,
)
from traffic_counter.plugin_ui.customtkinter_gui.frame_start import TabviewStart
from traffic_counter.plugin_ui.customtkinter_gui.frame_tracks import TracksFrame
from traffic_counter.plugin_ui.customtkinter_gui.frame_video_player import (
    FrameVideoPlayer,
)
from traffic_counter.plugin_ui.customtkinter_gui.frame_videos import FrameVideos
from traffic_counter.plugin_ui.customtkinter_gui.helpers import get_widget_position
from traffic_counter.plugin_ui.customtkinter_gui.messagebox import InfoBox

CANVAS: str = "Canvas"
VIDEO: str = "Video"


class ModifiedCTk(AbstractMainWindow, CTk):
    def __init__(
        self,
        viewmodel: ViewModel,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.protocol("WM_DELETE_WINDOW", self._ask_to_close)
        self._viewmodel: ViewModel = viewmodel
        self.introduce_to_viewmodel()

    def _ask_to_close(self) -> None:
        infobox = InfoBox(
            title="Close application",
            message="Do you want to close the application?",
            initial_position=get_widget_position(self),
            show_cancel=True,
        )
        if infobox.canceled:
            return
        self.quit()

    def introduce_to_viewmodel(self) -> None:
        self._viewmodel.set_window(self)

    def get_position(self, offset: tuple[float, float] = (0.5, 0.5)) -> tuple[int, int]:
        x, y = get_widget_position(self, offset=offset)
        return x, y

    def report_callback_exception(
        self, exc: BaseException, val: Any, tb: Any  # BaseExceptionGroup
    ) -> None:
        messages = gather_exception_messages(val)
        message = "\n".join(messages)
        logger().exception(messages, exc_info=True)
        InfoBox(message=message, title="Error", initial_position=self.get_position())


class TabviewInputFiles(CustomCTkTabview):
    def __init__(
        self,
        viewmodel: ViewModel,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._viewmodel = viewmodel
        self.TRACKS: str = "Tracks"
        self.VIDEOS: str = "Videos"
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self.add(self.TRACKS)
        self.frame_tracks = TracksFrame(
            master=self.tab(self.TRACKS), viewmodel=self._viewmodel
        )
        self.add(self.VIDEOS)
        self.frame_videos = FrameVideos(
            master=self.tab(self.VIDEOS), viewmodel=self._viewmodel
        )

    def _place_widgets(self) -> None:
        self.frame_tracks.pack(fill=tkinter.BOTH, expand=True)
        self.frame_videos.pack(fill=tkinter.BOTH, expand=True)
        self.set(self.TRACKS)


class FrameContent(CTkFrame):
    def __init__(
        self, master: Any, viewmodel: ViewModel, layers: Sequence[Layer], **kwargs: Any
    ) -> None:
        super().__init__(master=master, **kwargs)
        self.selected_video = None
        self._viewmodel = viewmodel
        self._layers = layers
        self._introduce_to_viewmodel()
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self):
        self._frame_track_plotting = FrameTrackPlotting(
            master=self,
            viewmodel=self._viewmodel,
            layers=self._layers,
        )
        self._frame_filter = FrameFilter(master=self, viewmodel=self._viewmodel)
        self._frame_canvas = FrameCanvas(
            master=self,
            viewmodel=self._viewmodel,
        )
        self._process_button = CTkButton(
            master=self,
            text="Process video",
            command=self._viewmodel.process_video,
            width=350,
            height=80,
            font=CTkFont(weight="bold", size=20),
        )

    def _place_widgets(self):
        self.grid_rowconfigure(0, weight=5)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=1, minsize=400)
        self._frame_canvas.grid(row=0, column=0, pady=PADY, sticky=STICKY)

    def _introduce_to_viewmodel(self):
        self._viewmodel.set_frame_content(self)

    def update_items(self) -> None:
        if self.selected_video:
            self.update_selected_items([self.selected_video])

    def update_selected_items(self, item_ids: list[str]):
        if not len(item_ids):
            return
        self.selected_video = item_ids[0]
        track_files = self._viewmodel.get_all_track_files()
        for track_file in track_files:
            if str(track_file).startswith(self.selected_video.rsplit(".")[0] + "_"):
                self.show_widgets(True)
                return
        self.show_widgets(False)

    def set_processed(self):
        self.grid_columnconfigure(1, weight=1, minsize=400)

        self._frame_canvas.grid(row=0, column=0, pady=PADY, sticky=STICKY)
        self._frame_track_plotting.grid(row=0, column=1, pady=PADY, sticky=STICKY)
        self._frame_filter.grid(row=1, column=0, pady=PADY, sticky=STICKY)
        self._process_button.grid_forget()

    def set_not_processed(self):
        self.grid_columnconfigure(1, weight=1, minsize=0)

        self._frame_canvas.grid(row=0, column=0, padx=PADX*5, pady=PADY*5, sticky=STICKY)
        self._frame_track_plotting.grid_forget()
        self._frame_filter.grid_forget()
        self._process_button.grid(row=1, column=0, pady=PADY, padx=PADX)

    def show_widgets(self, is_processed):
        if is_processed:
            self.set_processed()
        else:
            self.set_not_processed()


class FrameNavigation(EmbeddedCTkScrollableFrame):
    def __init__(self, master: Any, viewmodel: ViewModel, **kwargs: Any) -> None:
        super().__init__(master=master, **kwargs)
        self._viewmodel = viewmodel
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self._frame_project = TabviewProject(
            master=self,
            viewmodel=self._viewmodel,
            height=10
        )
        self._tabview_input_files = TabviewInputFiles(
            master=self, viewmodel=self._viewmodel
        )
        # self._frame_start = TabviewStart(master=self, viewmodel=self._viewmodel)
        # self._tabview_input_files = TabviewFiles(master=self, viewmodel=self._viewmodel)
        self._tabview_configuration = TabviewConfiguration(
            master=self, viewmodel=self._viewmodel
        )
        self._frame_analysis = TabviewAnalysis(master=self, viewmodel=self._viewmodel)

    def _place_widgets(self) -> None:
        self.grid_rowconfigure((1, 2), weight=1)
        self.grid_columnconfigure((0, 3), weight=0)
        # self._frame_start.grid(row=0, column=0, pady=PADY, sticky=STICKY)
        self._frame_project.grid(row=0, column=0, pady=PADY, sticky=STICKY)
        # self._tabview_input_files.grid(row=1, column=0, pady=PADY, sticky=STICKY)
        self._tabview_configuration.grid(row=1, column=0, pady=PADY, sticky=STICKY)
        self._frame_analysis.grid(row=2, column=0, pady=PADY, sticky=STICKY)


class TabviewContent(CustomCTkTabview):
    def __init__(
        self,
        viewmodel: ViewModel,
        layers: Sequence[Layer],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._viewmodel = viewmodel
        self._layers = layers
        self._get_widgets()
        self._place_widgets()

    def _get_widgets(self) -> None:
        self.add(CANVAS)
        self.frame_tracks = FrameContent(
            master=self.tab(CANVAS), viewmodel=self._viewmodel, layers=self._layers
        )
        self.add(VIDEO)
        self.frame_video = FrameVideoPlayer(
            master=self.tab(VIDEO), viewmodel=self._viewmodel
        )

    def _place_widgets(self) -> None:
        self.frame_tracks.pack(fill=tkinter.BOTH, expand=True)
        self.frame_video.pack(fill=tkinter.BOTH, expand=True)
        self.set(CANVAS)


class OTAnalyticsGui:
    def __init__(
        self,
        app: ModifiedCTk,
        view_model: ViewModel,
        layers: Sequence[Layer],
    ) -> None:
        self._viewmodel = view_model
        self._app = app
        self._layers = layers

    def start(self) -> None:
        self._show_gui()

    def _show_gui(self) -> None:
        set_appearance_mode("System")
        set_default_color_theme("blue")

        self._app.title("Traffic Counter")
        self._app.minsize(width=1024, height=768)

        self._get_widgets()
        self._place_widgets()
        self._app.after(0, lambda: self._app.state("zoomed"))
        self._app.mainloop()

    def _get_widgets(self) -> None:
        self._navigation = FrameNavigation(
            master=self._app,
            viewmodel=self._viewmodel,
            width=336,
        )
        self._content = TabviewContent(
            master=self._app, viewmodel=self._viewmodel, layers=self._layers
        )

    def _place_widgets(self) -> None:
        self._app.grid_columnconfigure(0, minsize=300, weight=0)
        self._app.grid_columnconfigure(1, weight=1)
        self._app.grid_rowconfigure(0, weight=1)
        self._navigation.grid(row=0, column=0, padx=PADX, pady=PADY, sticky=STICKY)
        self._content.grid(
            row=0,
            column=1,
            padx=PADX,
            pady=PADY,
            sticky=STICKY,
        )
