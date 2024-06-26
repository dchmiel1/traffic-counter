from typing import Any, Iterator, Sequence

from tqdm import tqdm

from traffic_counter.domain.progress import Progressbar, ProgressbarBuilder


class TqdmProgressBar(Progressbar):
    def __init__(self, sequence: Sequence, description: str, unit: str) -> None:
        self.__sequence = sequence
        self.__current_iterator = tqdm(self.__sequence)
        self.__description = description
        self.__unit = unit

    def __iter__(self) -> Iterator:
        self.__current_iterator = tqdm(
            iterable=self.__sequence,
            desc=self.__description,
            unit=self.__unit,
            total=len(self.__sequence),
        ).__iter__()
        return self

    def __next__(self) -> Any:
        return next(self.__current_iterator)


class TqdmBuilder(ProgressbarBuilder):
    def __call__(
        self, sequence: Sequence, description: str, unit: str
    ) -> TqdmProgressBar:
        return TqdmProgressBar(sequence, description, " " + unit)
