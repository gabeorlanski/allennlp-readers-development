"""
Could not think of a better name for this file...so went with its literal
purpose which is to be a really really crappy testing script.

It serves its purpose.
"""

import plac
from pathlib import Path

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

# Have to import even though we do not use it so that the registrable are
# registered.
from my_project.datasets import superglue


def debugReader(file_path: Path):
    reader: DatasetReader = DatasetReader.by_name('superglue_record')()
    test = list(reader.read(file_path))
    print(f"{len(test)} examples read from {file_path}")


@plac.annotations(
    dataset_file=plac.Annotation('Dataset file to read with the Reader.', abbrev='i', type=str)
)
def main(dataset_file):
    # Assuming for my sanity that data is stored in a directory called 'data'
    debugReader(Path('data', dataset_file))


if __name__ == '__main__':
    plac.call(main)
