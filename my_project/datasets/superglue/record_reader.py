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
import json

logger = logging.getLogger(__name__)

__all__ = ["RecordTaskReader"]


@DatasetReader.register("superglue-record")
class RecordTaskReader(DatasetReader):
    """
    Reader for Reading Comprehension with Commonsense Reasoning(ReCoRD) task from SuperGLUE. The
    task is detailed in the paper ReCoRD: Bridging the Gap between Human and Machine Commonsense
    Reading Comprehension (arxiv.org/pdf/1810.12885.pdf) by Zhang et al. Leaderboards and the
    official evaluation script for the ReCoRD task can be found sheng-z.github.io/ReCoRD-explorer/.

    The reader reads a JSON file in the format from
    sheng-z.github.io/ReCoRD-explorer/dataset-readme.txt
    

    # Parameters

    tokenizer: `Tokenizer`, optional
        The tokenizer class to use. Defaults to SpacyTokenizer

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.

    passage_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the passage if the length of passage exceeds this limit.

    question_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the question if the length of question exceeds this limit.

    raise_errors: `bool`, optional (default=`False`)
        If the reader should raise errors or just continue.

    kwargs: `Dict`
        Keyword arguments to be passed to the DatasetReader parent class constructor.

    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 raise_errors: bool = False,
                 **kwargs) -> None:
        """
        Initialize the RecordTaskReader.
        """
        super(RecordTaskReader, self).__init__(**kwargs)

        # Save the values passed to __init__ to protected attributes
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._passage_length_limit = passage_length_limit
        self._question_length_limit = question_length_limit
        self._raise_errors = raise_errors

    @overrides
    def _read(self, file_path: Union[Path, str]) -> Iterable[Instance]:
        # IF `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        # Read the 'data' key from the dataset
        logger.info(f"Reading '{file_path}'")
        with open(file_path) as fp:
            dataset = json.load(fp)['data']
        logger.info(f"Found {len(dataset)} examples from '{file_path}'")

        # Iterate through every example from the ReCoRD data file.
        for example in dataset:
            example: Dict
            # Get the data from the example dict. Wrap in a Try block in order
            # to log any errors that come up. If raise_errors is enabled, it
            # will raise them.
            try:
                # Get the passage dict from the example, it has text and entities
                passage_dict: Dict = example['passage']
                passage_text: str = passage_dict['text']

                # Entities are stored as a dict with the keys 'start' and 'end' for
                # their respective char indices.
                passage_entities = [self.get_span_from_text(passage_text, e['start'], e['end'])
                                    for e in passage_dict['entities']]
                logger.debug(f"Found {len(passage_entities)} entities in {example['id']}")

                # Get the queries from the example dict
                queries: List = example['qas']
            except KeyError as e:
                logger.error(f"{example['id']} raised error '{e}'")
                if self._raise_errors:
                    raise e
                continue


    @staticmethod
    def get_span_from_text(text: str,
                           start: int,
                           end: int) -> str:
        """
        Helper function to get a span from a string

        Args:
            text: `str`
                The source string
            start: `int`
                The starting index
            end: `int`
                The end index. It is INCLUSIVE.

        Returns: `str`
            The extracted string from text.

        """
        return ''.join(text[start:end])

    @overrides
    def text_to_instance(
            self,
            passage: List[str],
    ) -> Instance:
        """
        TODO: Needs to Return
            question_with_context: Dict[str, Dict[str, torch.LongTensor]]
            context_span: torch.IntTensor
            cls_index: Optional[torch.LongTensor]
            answer_span: Optional[torch.IntTensor]
            metadata: Optional[List[Dict[str, Any]]]
        """

        raise NotImplementedError(
            "'RecordTaskReader.text_to_instance' is not yet implemented"
        )
