from datetime import timedelta
from pathlib import Path
from typing import Iterable

import xlsxwriter
from pandas import DataFrame

from traffic_counter.application.analysis.traffic_counting import (
    LEVEL_CLASSIFICATION,
    LEVEL_END_TIME,
    LEVEL_FLOW,
    LEVEL_FROM_SECTION,
    LEVEL_START_TIME,
    LEVEL_TO_SECTION,
    AddSectionInformation,
    Count,
    Exporter,
    ExporterFactory,
    FillEmptyCount,
    Tag,
    create_flow_tag,
    create_mode_tag,
    create_timeslot_tag,
)
from traffic_counter.application.analysis.traffic_counting_specification import (
    ExportFormat,
    ExportSpecificationDto,
)
from traffic_counter.application.logger import logger


class CsvExport(Exporter):
    def __init__(self, output_file: str) -> None:
        self._output_file = output_file

    def export(self, counts: Count) -> None:
        logger().info(f"Exporting counts to {self._output_file}")
        dataframe = self.__create_data_frame(counts)
        if dataframe.empty:
            logger().info("Nothing to count.")
            return
        dataframe = self._set_column_order(dataframe)
        dataframe = dataframe.sort_values(
            # by=[LEVEL_START_TIME, LEVEL_END_TIME, LEVEL_CLASSIFICATION]
            by=[LEVEL_CLASSIFICATION]
        )
        dataframe.to_csv(self.__create_path(), index=False)
        logger().info(f"Counts saved at {self._output_file}")

    @staticmethod
    def _set_column_order(dataframe: DataFrame) -> DataFrame:
        desired_columns_order = [
            # LEVEL_START_TIME,
            # LEVELT_END_TIME,
            LEVEL_CLASSIFICATION,
            LEVEL_FLOW,
            LEVEL_FROM_SECTION,
            LEVEL_TO_SECTION,
        ]
        dataframe = dataframe[
            desired_columns_order
            + [col for col in dataframe.columns if col not in desired_columns_order]
        ]

        return dataframe

    @staticmethod
    def __create_data_frame(counts: Count) -> DataFrame:
        transformed = counts.to_dict()
        indexed: list[dict] = []
        for key, value in transformed.items():
            result_dict: dict = key.as_dict()
            result_dict["count"] = value
            indexed.append(result_dict)
        return DataFrame.from_dict(indexed)

    def __create_path(self) -> Path:
        fixed_file_ending = (
            self._output_file
            if self._output_file.lower().endswith(".csv")
            else self._output_file + ".csv"
        )
        path = Path(fixed_file_ending)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class ODMatrixExporter(Exporter):
    def __init__(self, output_file):
        self._output_file = output_file

    def export(self, counts: Count) -> None:
        logger().info(f"Exporting counts to {self._output_file}")
        origins, destinations = self.__create_records(counts)
        if len(origins) == 0 or len(destinations) == 0:
            logger().info("Nothing to count.")
            return
        self._to_xlsx(origins, destinations)
        logger().info(f"Counts saved at {self._output_file}")

    def _to_xlsx(self, origins: list, destinations: dict):
        workbook = xlsxwriter.Workbook(self.__create_path())
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, "O\D")
        destinations = dict(sorted(destinations.items()))

        for i, o in enumerate(origins):
            worksheet.write(i + 1, 0, o)
        for j, (d, counts) in enumerate(destinations.items()):
            worksheet.write(0, j + 1, d)
            for k in range(len(origins)):
                worksheet.write(k + 1, j + 1, counts[k])
        workbook.close()

    @staticmethod
    def __create_records(counts: Count) -> DataFrame:
        transformed = counts.to_dict()
        origins = []
        destinations = {}
        for key, value in transformed.items():
            result_dict: dict = key.as_dict()
            o = result_dict["from section"]
            d = result_dict["to section"]
            if o not in origins:
                origins.append(o)
            if d not in destinations:
                destinations[d] = [0 for _ in range(len(transformed.keys()))]
            destinations[d][origins.index(o)] += value
        return origins, destinations



    def __create_path(self) -> Path:
        fixed_file_ending = (
            self._output_file
            if self._output_file.lower().endswith(".xlsx")
            else self._output_file + ".xlsx"
        )
        path = Path(fixed_file_ending)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path



class SimpleExporterFactory(ExporterFactory):
    def __init__(self) -> None:
        self._formats = {
            ExportFormat("CSV", ".csv"): lambda output_file: CsvExport(output_file),
            ExportFormat("XLSX", ".xlsx"): lambda output_file: ODMatrixExporter(
                output_file
            ),
        }
        self._factories = {
            format.name: factory for format, factory in self._formats.items()
        }

    def get_supported_formats(self) -> Iterable[ExportFormat]:
        return self._formats.keys()

    def create_exporter(self, specification: ExportSpecificationDto) -> Exporter:
        return self._factories[specification.format](specification.output_file)


class TagExploder:
    """
    This class creates all combinations of tags for a given ExportSpecificationDto.
    The resulting tags are a cross product of the flows, the modes and the time
    intervals. The list of tags can then be used as the maximum set of tags in the
    export.
    """

    def __init__(self, specification: ExportSpecificationDto):
        self._specification = specification

    def explode(self) -> list[Tag]:
        tags = []
        # start_without_seconds = (
        #     self._specification.counting_specification.start.replace(
        #         second=0, microsecond=0
        #     )
        # )
        # maximum = self._specification.counting_specification.end - start_without_seconds
        # duration = int(maximum.total_seconds())
        # interval = self._specification.counting_specification.interval_in_minutes * 60
        for flow in self._specification.flow_name_info:
            for mode in self._specification.counting_specification.modes:
                # for delta in range(0, duration, interval):
                #     offset = timedelta(seconds=delta)
                #     start = start_without_seconds + offset
                #     interval_time = timedelta(seconds=interval)
                #     tag = (
                #         create_flow_tag(flow.name)
                #         .combine(create_mode_tag(mode))
                #         .combine(create_timeslot_tag(start, interval_time))
                #     )
                #     tags.append(tag)
                tag = (
                    create_flow_tag(flow.name)
                    .combine(create_mode_tag(mode))
                )
                tags.append(tag)
        return tags

class FillZerosExporter(Exporter):
    def __init__(self, other: Exporter, tag_exploder: TagExploder) -> None:
        self._other = other
        self._tag_exploder = tag_exploder

    def export(self, counts: Count) -> None:
        tags = self._tag_exploder.explode()
        self._other.export(FillEmptyCount(counts, tags))


class FillZerosExporterFactory(ExporterFactory):
    def __init__(self, other: ExporterFactory) -> None:
        self.other = other

    def get_supported_formats(self) -> Iterable[ExportFormat]:
        return self.other.get_supported_formats()

    def create_exporter(self, specification: ExportSpecificationDto) -> Exporter:
        return FillZerosExporter(
            self.other.create_exporter(specification),
            TagExploder(specification),
        )


class AddSectionInformationExporter(Exporter):
    def __init__(self, other: Exporter, specification: ExportSpecificationDto) -> None:
        self._other = other
        self._specification = specification

    def export(self, counts: Count) -> None:
        flow_info_dict = {
            flow_dto.name: flow_dto for flow_dto in self._specification.flow_name_info
        }
        self._other.export(AddSectionInformation(counts, flow_info_dict))


class AddSectionInformationExporterFactory(ExporterFactory):
    def __init__(self, other: ExporterFactory) -> None:
        self.other = other

    def get_supported_formats(self) -> Iterable[ExportFormat]:
        return self.other.get_supported_formats()

    def create_exporter(self, specification: ExportSpecificationDto) -> Exporter:
        return AddSectionInformationExporter(
            self.other.create_exporter(specification), specification
        )
