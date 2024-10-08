from typing import Any

from customtkinter import CTkLabel, CTkCheckBox, CTkFrame, CTkFont

from traffic_counter.plugin_video_processing.process import (
    BOT_SORT_NAME,
    BOT_SORT_PLUS_NAME,
    CO_DETR_NAME,
    DEEP_OC_SORT_NAME,
    RT_DETR_NAME,
    SMILETRACK_NAME,
    YOLOV6_NAME,
)

from traffic_counter.plugin_ui.customtkinter_gui.constants import PADX, PADY, STICKY
from traffic_counter.plugin_ui.customtkinter_gui.toplevel_template import (
    FrameContent,
    ToplevelTemplate,
)


class CancelProcessing(Exception):
    pass


class DetectorNotSelected(Exception):
    def __init__(self):
        super().__init__("Detector not selected")


class TrackerNotSelected(Exception):
    def __init__(self):
        super().__init__("Tracker not selected")


class FrameSetAlgorithms(CTkFrame):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._get_widgets()
        self._place_widgets()

    def check_yolo(self):
        self.checkbox_rtdetr.deselect()
        self.checkbox_codetr.deselect()

    def check_rt_detr(self):
        self.checkbox_yolo.deselect()
        self.checkbox_codetr.deselect()

    def check_co_detr(self):
        self.checkbox_yolo.deselect()
        self.checkbox_rtdetr.deselect()

    def check_bot_sort(self):
        self.checkbox_smiletrack.deselect()
        self.checkbox_deep_oc_sort.deselect()
        self.checkbox_botsort_plus.deselect()

    def check_smiletrack(self):
        self.checkbox_botsort.deselect()
        self.checkbox_deep_oc_sort.deselect()
        self.checkbox_botsort_plus.deselect()

    def check_deep_oc_sort(self):
        self.checkbox_botsort.deselect()
        self.checkbox_smiletrack.deselect()
        self.checkbox_botsort_plus.deselect()

    def check_botsort_plus(self):
        self.checkbox_botsort.deselect()
        self.checkbox_smiletrack.deselect()
        self.checkbox_deep_oc_sort.deselect()

    def _get_widgets(self) -> None:
        self.label_detector = CTkLabel(master=self, text="Detector")
        self.checkbox_yolo = CTkCheckBox(
            master=self, text=YOLOV6_NAME, command=self.check_yolo, corner_radius=50
        )
        self.checkbox_rtdetr = CTkCheckBox(
            master=self, text=RT_DETR_NAME, command=self.check_rt_detr, corner_radius=50
        )
        self.checkbox_codetr = CTkCheckBox(
            master=self, text=CO_DETR_NAME, command=self.check_co_detr, corner_radius=50
        )

        self.label_tracker = CTkLabel(master=self, text="Tracker")
        self.checkbox_botsort = CTkCheckBox(
            master=self,
            text=BOT_SORT_NAME,
            command=self.check_bot_sort,
            corner_radius=50,
        )
        self.checkbox_smiletrack = CTkCheckBox(
            master=self,
            text=SMILETRACK_NAME,
            command=self.check_smiletrack,
            corner_radius=50,
        )
        self.checkbox_deep_oc_sort = CTkCheckBox(
            master=self,
            text=DEEP_OC_SORT_NAME,
            command=self.check_deep_oc_sort,
            corner_radius=50,
        )
        self.checkbox_botsort_plus = CTkCheckBox(
            master=self,
            text=BOT_SORT_PLUS_NAME,
            command=self.check_botsort_plus,
            corner_radius=50,
        )

        self.det_checkboxes = [
            self.checkbox_yolo,
            self.checkbox_rtdetr,
            self.checkbox_codetr,
        ]

        self.tracker_checkboxes = [
            self.checkbox_botsort,
            self.checkbox_smiletrack,
            self.checkbox_deep_oc_sort,
            self.checkbox_botsort_plus,
        ]

        self.save_vid_checkbox = CTkCheckBox(
            master=self,
            text="Save processed video",
            corner_radius=0,
            font=CTkFont(size=14),
            checkbox_width=18,
            checkbox_height=18,
        )

    def _place_widgets(self) -> None:
        self.grid_columnconfigure((0, 1), pad=PADX * 3)
        self.grid_rowconfigure((5), minsize=20)
        self.label_detector.grid(
            row=0, column=0, padx=PADX * 3, pady=PADY * 3, sticky=STICKY
        )
        self.checkbox_yolo.grid(
            row=1, column=0, padx=PADX * 3, pady=PADY, sticky=STICKY
        )
        self.checkbox_rtdetr.grid(
            row=2, column=0, padx=PADX * 3, pady=PADY, sticky=STICKY
        )
        self.checkbox_codetr.grid(
            row=3, column=0, padx=PADX * 3, pady=PADY, sticky=STICKY
        )

        self.label_tracker.grid(
            row=0, column=1, padx=PADX * 3, pady=PADY, sticky=STICKY
        )
        self.checkbox_botsort.grid(
            row=1, column=1, padx=PADX * 3, pady=PADY, sticky=STICKY
        )
        self.checkbox_smiletrack.grid(
            row=2, column=1, padx=PADX * 3, pady=PADY, sticky=STICKY
        )
        self.checkbox_deep_oc_sort.grid(
            row=3, column=1, padx=PADX * 3, pady=PADY, sticky=STICKY
        )
        self.checkbox_botsort_plus.grid(
            row=4, column=1, padx=PADX * 3, pady=PADY, sticky=STICKY
        )

        self.save_vid_checkbox.grid(
            row=6, column=0, padx=PADX, pady=PADY * 3, sticky=STICKY
        )

    def set_focus(self):
        pass

    def get_selected_detector(self):
        for checkbox in self.det_checkboxes:
            if checkbox.get():
                return checkbox._text

    def get_selected_tracker(self):
        for checkbox in self.tracker_checkboxes:
            if checkbox.get():
                return checkbox._text

    def is_save_video_checked(self):
        return self.save_vid_checkbox.get()


class VideoProcessingChoiceWindow(ToplevelTemplate):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _create_frame_content(self, master: Any) -> FrameContent:
        return FrameSetAlgorithms(master=master)

    def _on_ok(self, event: Any = None) -> None:
        self.det = self._frame_content.get_selected_detector()
        if self.det is None:
            raise DetectorNotSelected()
        self.tracker = self._frame_content.get_selected_tracker()
        if self.tracker is None:
            raise TrackerNotSelected()
        self.save_processed_video = self._frame_content.is_save_video_checked()

        self._close()

    def get_data(self) -> dict:
        self.wait_window()
        if self._canceled:
            raise CancelProcessing()
        return self.det, self.tracker, self.save_processed_video
