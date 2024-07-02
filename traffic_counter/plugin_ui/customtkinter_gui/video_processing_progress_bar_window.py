from typing import Any

from customtkinter import CTkProgressBar, CTkToplevel, CTkLabel, CTkFont


class VideoProcessingProgressBarWindow(CTkToplevel):
    def __init__(
        self, title: str, initial_position: tuple[int, int], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.title(title)
        self._get_widgets()
        self._place_widgets()
        self._set_initial_position(initial_position)
        self._set_focus()

    def _get_widgets(self):
        self.progressbar = CTkProgressBar(self, height=30, width=300, corner_radius=0)
        self.progresslabel = CTkLabel(self, height=50, width=300, text="0%", font=CTkFont(size=18))
        self.progressbar.set(0)

    def _place_widgets(self):
        self.progressbar.pack(padx=20, pady=20)
        self.progresslabel.pack(padx=20, pady=0)

    def _set_initial_position(self, initial_position: tuple[int, int]) -> None:
        x, y = initial_position
        x0 = x - (self.winfo_width() // 2)
        y0 = y - (self.winfo_height() // 2)
        self.geometry(f"+{x0+10}+{y0+10}")

    def _set_focus(self) -> None:
        self.attributes("-topmost", 1)
        self.after(0, lambda: self.lift())
        self.after(0, lambda: self.tk.call('focus', self._w))

    def update(self, value):
        if value == 1:
            self.destroy()
            return
    
        self.progresslabel.configure(text=f"{round(value*100)}%")
        self.progressbar.set(value)
