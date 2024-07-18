from pathlib import Path

from traffic_counter.application.datastore import Datastore
from traffic_counter.application.parser.config_parser import ConfigParser
from traffic_counter.application.use_cases.track_repository import GetAllTrackFiles


class MissingDate(Exception):
    pass


class SaveOtconfig:
    def __init__(
        self,
        datastore: Datastore,
        get_all_track_files: GetAllTrackFiles,
        config_parser: ConfigParser,
    ) -> None:
        self._datastore = datastore
        self.get_all_track_files = get_all_track_files
        self._config_parser = config_parser

    def __call__(self, file: Path) -> None:
        # if self._datastore.project.start_date:
        self._config_parser.serialize(
            project=self._datastore.project,
            video_files=self._datastore.get_all_videos(),
            track_files=self.get_all_track_files(),
            sections=self._datastore.get_all_sections(),
            flows=self._datastore.get_all_flows(),
            file=file,
        )
        # else:
        #     raise MissingDate()
