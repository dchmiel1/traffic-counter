from typing import Callable, Iterable, Sequence

from traffic_counter.domain.event import Event
from traffic_counter.domain.intersect import IntersectParallelizationStrategy
from traffic_counter.domain.section import Section
from traffic_counter.domain.track_dataset import TrackDataset


class SequentialIntersect(IntersectParallelizationStrategy):
    """Executes the intersection of tracks and sections in sequential order."""

    @property
    def num_processes(self) -> int:
        return 1

    def execute(
        self,
        intersect: Callable[[TrackDataset, Iterable[Section]], Iterable[Event]],
        tasks: Sequence[tuple[TrackDataset, Iterable[Section]]],
    ) -> list[Event]:
        events: list[Event] = []
        for task in tasks:
            track_dataset, sections = task
            events.extend(intersect(track_dataset, sections))
        return events

    def set_num_processes(self, value: int) -> None:
        pass
