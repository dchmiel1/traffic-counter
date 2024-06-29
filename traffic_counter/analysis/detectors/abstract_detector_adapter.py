from abc import ABC, abstractmethod


class DetectorAdapter(ABC):
    @abstractmethod
    def _convert_dets(self):
        pass

    @abstractmethod
    def detect(im):
        pass
