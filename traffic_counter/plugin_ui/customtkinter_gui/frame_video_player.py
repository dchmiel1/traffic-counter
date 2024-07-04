from typing import Any

import cv2
import PIL.Image, PIL.ImageTk
from customtkinter import CTkButton, CTkSlider, CTkLabel

from traffic_counter.adapter_ui.view_model import ViewModel
from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    EmbeddedCTkFrame,
)


class VideoDisplay(CTkLabel):

    def __init__(self, master):
        super().__init__(master, text="")
        self.master = master
        self.paused = True
        self.delay_ms = 10
        self.vid = None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def _calculate_display_size(self):
        if not self.vid:
            return None

        if (self.winfo_width(), self.winfo_height()) == (1, 1):
            return 1, 1

        width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        factor_w = self.winfo_width() / width
        factor_h = self.winfo_height() / height
        factor = min(factor_w, factor_h)
        return int(width * factor), int(height * factor)

    def _get_frame(self):
        if self.vid.isOpened():
            ret, self.frame = self.vid.read()
            if ret:
                frame = cv2.resize(self.frame, self._calculate_display_size())
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                self.pause()
                self.seek(0)
                if self.video_ended_handler:
                    self.video_ended_handler()
                return (ret, None)
        else:
            return (ret, None)

    def _display_frame(self, frame):
        self.image = PIL.Image.fromarray(frame)
        self.photo = PIL.ImageTk.PhotoImage(image=self.image)
        self.configure(image=self.photo)
        # self.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        if self.progress_slider_updater:
            self.progress_slider_updater(self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    def _update_widget(self):
        if self.paused:
            return

        ret, frame = self._get_frame()
        if ret:
            self._display_frame(frame)

        self.master.after(self.delay_ms, self._update_widget)

    def _display_one_frame(self):
        ret, frame = self._get_frame()
        if ret:
            self._display_frame(frame)

    def pause(self):
        self.paused = True

    def play(self):
        self.paused = False
        self._update_widget()

    def load_video(self, video_source):
        if self.vid and self.vid.isOpened():
            self.vid.release()

        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self._display_one_frame()
        if self.progress_slider_initializer:
            self.progress_slider_initializer(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def seek(self, frame):
        if self.vid and self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
            if self.paused:
                self._display_one_frame()

    def set_progress_slider_updater(self, updater):
        self.progress_slider_updater = updater

    def set_progress_slider_initializer(self, initializer):
        self.progress_slider_initializer = initializer

    def set_video_ended_handler(self, handler):
        self.video_ended_handler = handler

    def on_window_resize(self, event):
        if self.vid and self.frame is not None:
            frame = cv2.resize(self.frame, self._calculate_display_size())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._display_frame(frame)


class FrameVideoPlayer(EmbeddedCTkFrame):
    def __init__(self, viewmodel: ViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.video_file = None
        self._viewmodel = viewmodel
        self._get_widgets()
        self._place_widgets()
        self._activate_widgets()
        self.introduce_to_viewmodel()

    def introduce_to_viewmodel(self) -> None:
        self._viewmodel.set_video_player(self)

    def _open_video(self) -> None:
        if self.video_file:
            self.vid_player.load_video(self.video_file)
            self.progress_slider.set(1)
            self.btn_play_pause.configure(text="Play ►")

    def _seek(self, value) -> None:
        if self.video_file:
            self.vid_player.seek(int(value))
        else:
            self.progress_slider.set(0)

    def _play_pause(self) -> None:
        if self.video_file:
            if self.vid_player.paused:
                self.vid_player.play()
                self.btn_play_pause.configure(text="Pause ||")
            else:
                self.vid_player.pause()
                self.btn_play_pause.configure(text="Play ►")

    def _init_slider(self, frames) -> None:
        self.progress_slider.configure(from_=1, to=frames, number_of_steps=frames)

    def _update_slider(self, frame) -> None:
        self.progress_slider.set(frame)

    def _video_ended_handler(self) -> None:
        self.btn_play_pause.configure(text="Play ►")
        self.progress_slider.set(1)

    def _get_widgets(self) -> None:
        self.vid_player = VideoDisplay(master=self)

        self.progress_slider = CTkSlider(
            master=self, from_=1, to=100, number_of_steps=100, command=self._seek
        )
        self.btn_play_pause = CTkButton(
            master=self, text="Play ►", command=self._play_pause
        )

    def _place_widgets(self) -> None:
        self.vid_player.pack(expand=True, fill="both", padx=80, pady=80)
        self.progress_slider.pack(fill="both", padx=80, pady=20)
        self.btn_play_pause.pack(pady=10)

    def _activate_widgets(self) -> None:
        self.vid_player.set_progress_slider_initializer(self._init_slider)
        self.vid_player.set_progress_slider_updater(self._update_slider)
        self.vid_player.set_video_ended_handler(self._video_ended_handler)
        self.vid_player.bind("<Configure>", self.vid_player.on_window_resize)
        self.progress_slider.set(1)

    def update_items(self) -> None:
        pass

    def update_selected_items(self, item_ids: list[str]):
        self.video_file = item_ids[0].replace("\\", "/")
        self._open_video()
