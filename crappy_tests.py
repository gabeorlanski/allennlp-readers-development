"""
Could not think of a better name for this file...so went with its literal
purpose which is to be a really really crappy testing script.

It serves its purpose.
"""

import plac
from pathlib import Path

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from src.util.log_util import setupLoggers
import logging
# Have to import even though we do not use it so that the registrable are
# registered.
from src.readers import superglue


def debugReader(file_path: Path, main_logger: logging.Logger):
    reader_name = 'superglue_record'
    main_logger.info(f"Reading '{file_path}' with reader '{reader_name}'")
    reader: DatasetReader = DatasetReader.by_name(reader_name)()
    test = list(reader.read(file_path))
    print(f"{len(test)} examples read from {file_path}")


@plac.annotations(
    test_name=plac.Annotation('Test to run for debugging', choices=['Reader']),
    reader_file=plac.Annotation('Dataset file to read with the Reader.', kind='option', type=str),
)
def main(test_name: str, reader_file: str = None) -> None:
    main_logger, issue_logger = setupLoggers(test_name, debug=True, verbose=True)
    main_logger.info(f"Starting {test_name}")

    if test_name == "Reader":
        assert reader_file is not None

        # Assuming for my sanity that data is stored in a directory called 'data'
        debugReader(Path('data', reader_file),main_logger)


if __name__ == '__main__':
    plac.call(main)
