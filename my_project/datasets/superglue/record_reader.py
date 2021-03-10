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
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer

logger = logging.getLogger(__name__)

__all__ = ["RecordTaskReader"]


@DatasetReader.register("superglue-record")
class RecordTaskReader(DatasetReader):
    """
    Reader for Reading Comprehension with Commonsense Reasoning(ReCoRD) task from SuperGLUE. The
    task is detailed in the paper ReCoRD: Bridging the Gap between Human and Machine Commonsense
    Reading Comprehension (arxiv.org/pdf/1810.12885.pdf) by Zhang et al. Leaderboards and the
    official evaluation script for the ReCoRD task can be found sheng-z.github.io/ReCoRD-explorer/.

    The reader reads a JSON file in the format from sheng-z.github.io/ReCoRD-explorer/dataset-readme.txt

    # Parameters

    tokenizer: `Tokenizer`
        The tokenizer class to use. Defaults to SpacyTokenizer

    kwargs: `Dict`
        Keyword arguments to be passed to the DatasetReader parent class constructor.

    """

    def __init__(self, tokenizer: Tokenizer = None, **kwargs) -> None:
        """
        Initialize the RecordTaskReader.
        """
        super(RecordTaskReader, self).__init__(**kwargs)

        # Load either the passed tokenizer or initialize a spacy tokenizer.
        self._tokenizer = tokenizer or SpacyTokenizer()

    @overrides
    def _read(self, file_path: Union[Path, str]) -> Iterable[Instance]:
        raise NotImplementedError("'RecordTaskReader._read' is not yet implemented")

    def text_to_instance(
            self,
            passage: List[str],
    ) -> Instance:

        """
        TODO: Needs to Return
            question_with_context: Dict[str, Dict[str, torch.LongTensor]],
            context_span: torch.IntTensor
            cls_index: Optional[torch.LongTensor]
            answer_span: Optional[torch.IntTensor]
            metadata: Optional[List[Dict[str, Any]]]
        """

        raise NotImplementedError(
            "'RecordTaskReader.text_to_instance' is not yet implemented"
        )
