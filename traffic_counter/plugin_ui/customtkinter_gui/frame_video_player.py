from typing import Any

from customtkinter import CTkButton, CTkSlider
from tkinter.filedialog import askopenfilename
from tkVideoPlayer import TkinterVideo

from traffic_counter.plugin_ui.customtkinter_gui.custom_containers import (
    EmbeddedCTkFrame,
)


class FrameVideoPlayer(EmbeddedCTkFrame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.video_file = None
        self._get_widgets()
        self._place_widgets()
        self._activate_widgets()

    def _open_video(self) -> None:
        self.vid_player.stop()
        self.video_file = askopenfilename(
            filetypes=[
                ("Video", ["*.mp4", "*.avi", "*.mov", "*.mkv", "*gif"]),
                ("All Files", "*.*"),
            ]
        )
        if self.video_file:
            try:
                self.vid_player.load(self.video_file)
                self.vid_player.play()
                self.progress_slider.set(-1)
                self.btn_play_pause.configure(text="Pause ||")
            except:
                print("Unable to load the file")

    def _update_duration(self, event) -> None:
        try:
            duration = int(self.vid_player.video_info()["duration"])
            self.progress_slider.configure(
                from_=-1, to=duration, number_of_steps=duration
            )
        except:
            pass

    def _seek(self, value) -> None:
        if self.video_file:
            try:
                self.vid_player.seek(int(value))
                self.vid_player.play()
                self.vid_player.after(50, self.vid_player.pause)
                self.btn_play_pause.configure(text="Play ►")
            except:
                pass

    def _update_scale(self, event) -> None:
        try:
            self.progress_slider.set(int(self.vid_player.current_duration()))
        except:
            pass

    def _play_pause(self) -> None:
        if self.video_file:
            if self.vid_player.is_paused():
                self.vid_player.play()
                self.btn_play_pause.configure(text="Pause ||")

            else:
                self.vid_player.pause()
                self.btn_play_pause.configure(text="Play ►")

    def _video_ended(self, event) -> None:
        self.btn_play_pause.configure(text="Play ►")
        self.progress_slider.set(-1)

    def _get_widgets(self) -> None:
        self.btn_open_video = CTkButton(
            master=self, text="Open Video", corner_radius=8, command=self._open_video
        )
        self.vid_player = TkinterVideo(
            master=self,
            scaled=True,
            keep_aspect=True,
            consistant_frame_rate=True,
            bg="black",
        )
        self.progress_slider = CTkSlider(
            master=self, from_=-1, to=1, number_of_steps=1, command=self._seek
        )
        self.btn_play_pause = CTkButton(
            master=self, text="Play ►", command=self._play_pause
        )

    def _place_widgets(self) -> None:
        self.vid_player.pack(expand=True, fill="both", padx=10, pady=10)
        self.btn_open_video.pack(pady=10, padx=10)
        self.progress_slider.pack(fill="both", padx=10, pady=10)
        self.btn_play_pause.pack(pady=10)

    def _activate_widgets(self) -> None:
        self.vid_player.set_resampling_method(1)
        self.vid_player.bind("<<Duration>>", self._update_duration)
        self.vid_player.bind("<<SecondChanged>>", self._update_scale)
        self.vid_player.bind("<<Ended>>", self._video_ended)
        self.progress_slider.set(-1)
