"""
Dataset reader for SuperGLUE's Reading Comprehension with Commonsense Reasoning task (Zhang Et
al. 2018).

Reader Implemented by Gabriel Orlanski
"""

from typing import Dict, List, Optional, Sequence, Iterable, Union
import itertools
import logging
import warnings
from pathlib import Path

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)

__all__ = [
    "RecordTaskReader"
]


@DatasetReader.register("superglue-record")
class RecordTaskReader(DatasetReader):
    def __init__(
            self,
    ) -> None:
        raise NotImplementedError("'RecordTaskReader.__init__' is not yet implemented")

    @overrides
    def _read(self, file_path: Union[Path, str]) -> Iterable[Instance]:
        raise NotImplementedError("'RecordTaskReader._read' is not yet implemented")

    def text_to_instance(  # type: ignore
            self,
            tokens: List[Token],
            pos_tags: List[str] = None,
            chunk_tags: List[str] = None,
            ner_tags: List[str] = None,
    ) -> Instance:
        raise NotImplementedError("'RecordTaskReader.text_to_instance' is not yet implemented")
