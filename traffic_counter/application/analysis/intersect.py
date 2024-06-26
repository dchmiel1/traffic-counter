from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, Mapping

from traffic_counter.domain.event import Event
from traffic_counter.domain.geometry import RelativeOffsetCoordinate
from traffic_counter.domain.section import Section, SectionId
from traffic_counter.domain.track import TrackId
from traffic_counter.domain.types import EventType


class IntersectionError(Exception):
    pass


class RunIntersect(ABC):
    """
    Interface defining the use case to intersect the given tracks with the given
    sections.
    """

    @abstractmethod
    def __call__(self, sections: Iterable[Section]) -> list[Event]:
        raise NotImplementedError


class TracksIntersectingSections(ABC):
    @abstractmethod
    def __call__(self, sections: Iterable[Section]) -> dict[SectionId, set[TrackId]]:
        raise NotImplementedError


def group_sections_by_offset(
    sections: Iterable[Section], offset_type: EventType = EventType.SECTION_ENTER
) -> Mapping[RelativeOffsetCoordinate, list[Section]]:
    grouped_sections: dict[RelativeOffsetCoordinate, list[Section]] = defaultdict(list)
    for section in sections:
        offset = section.get_offset(offset_type)
        grouped_sections[offset].append(section)
    return grouped_sections
