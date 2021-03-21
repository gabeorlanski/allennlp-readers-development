"""
Dataset reader for SuperGLUE's Reading Comprehension with Commonsense Reasoning task (Zhang Et
al. 2018).

Reader Implemented by Gabriel Orlanski
"""

from typing import Dict, List, Optional, Sequence, Iterable, Union, Tuple, Any
import itertools
import logging
import warnings
from pathlib import Path

from overrides import overrides
from src.util.log_util import getBothLoggers

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
import json

logger, _ = getBothLoggers()

__all__ = ['RecordTaskReader']


@DatasetReader.register("superglue_record")
class RecordTaskReader(DatasetReader):
    # TODO: Update and improve this docstring
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

    # TODO: Expand on init args and add correct tokenizers
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 length_limit: int = 384,
                 question_length_limit: int = 64,
                 stride: int = 128,
                 raise_errors: bool = False,
                 **kwargs) -> None:
        """
        Initialize the RecordTaskReader.
        """
        super(RecordTaskReader, self).__init__(**kwargs)

        # Save the values passed to __init__ to protected attributes
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._length_limit = length_limit
        self._query_len_limit = question_length_limit
        self._stride = stride
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

        # Keep track of certain stats while reading the file
        # examples_multiple_instance_count: The number of questions with more than
        #   one instance. Can happen because there is multiple queries for a
        #   single passage.
        # passages_yielded: The total number of instances found/yielded.
        examples_multiple_instance_count = 0
        passages_yielded = 0

        # Iterate through every example from the ReCoRD data file.
        for example in dataset:
            example: Dict

            # Get the list of instances for the current example
            instances_for_example = self.get_instances_from_example(example)

            # Keep track of number of instances for this specific example that
            # have been yielded. Since it instances_for_example is a generator, we
            # do not know its length. To address this, we create an counter int.
            instance_count = 0

            # Iterate through the instances and yield them.
            for instance in instances_for_example:
                yield instance
                instance_count += 1

            # Check if there was more than one instance for this example. If
            # there was we increase examples_multiple_instance_count by 1.
            # Otherwise we increase by 0.
            examples_multiple_instance_count += 1 if instance_count > 1 else 0

            passages_yielded += 1

        # Log pertinent information.
        if passages_yielded:
            logger.info(f"{examples_multiple_instance_count}/{passages_yielded} "
                        f"({examples_multiple_instance_count / passages_yielded * 100:.2f}%) "
                        f"examples had more than one instance")
        else:
            logger.warning(f"Could not find any instances in '{file_path}'")

    def get_instances_from_example(self, example: Dict) -> Iterable[Instance]:
        """
        Helper function to get instances from an example.

        Much of this comes from `transformer_squad.make_instances`
        Args:
            example: `Dict[str,Any]`
                The example dict.

        Yields: `Iterable[Instance]`
            The instances for each example
        """
        # Get the passage dict from the example, it has text and
        # entities
        example_id: str = example['id']
        passage_dict: Dict = example['passage']
        passage_text: str = passage_dict['text']

        # Tokenize the passage
        tokenized_passage: Iterable[Token] = self.tokenize_str(passage_text)

        # TODO: Determine what to do with entities. Superglue marks them
        #   explicitly as input (https://arxiv.org/pdf/1905.00537.pdf)

        # # Entities are stored as a dict with the keys 'start' and 'end'
        # # for their respective char indices. We use the helper function
        # # tokenize_slice to get these spans and tokenize them.
        # passage_entities = [self.tokenize_slice(passage_text, start, end)
        #                     for start, end in passage_dict['entities']]
        # logger.debug(f"Found {len(passage_entities)} entities in {example['id']}")

        # Get the queries from the example dict
        queries: List = example['qas']
        logger.debug(f"{len(queries)} queries for example {example_id}")

        # Tokenize and get the context windows for each queries
        for query in queries:
            query: Dict

            # Create the additional metadata dict that will be passed w/ extra
            # data for each query. We store the question & query ids, all
            # answers, and other data following `transformer_qa`.
            additional_metadata = {
                "id"        : query['id'],
                "example_id": example_id,
            }

            # Tokenize, and truncate, the query based on the max set in
            # `__init__`
            tokenized_query = self.tokenize_str(query['query'])[:self._query_len_limit]

            # Calculate the remaining space for the context w/ the length of the
            # question special tokens.

    def tokenize_slice(self,
                       text: str,
                       start: int,
                       end: int) -> Iterable[Token]:
        """
        Get + tokenize a span from a source text.

        *Originally from the `transformer_squad.py`*

        Args:
            text: `str`
                The text to draw from.
            start: `int`
                The start index for the span.
            end: `int`
                The end index for the span. Assumed that this is inclusive.

        Returns: `Iterable[Token]`
            List of tokens for the retrieved span.
        """
        text_to_tokenize = text[start:end]

        # Check if this is the start of the text. If the start is >= 0, check
        # for a preceding space. If it exists, then we need to tokenize a
        # special way because of a bug with RoBERTa tokenizer.
        if start - 1 >= 0 and text[start - 1].isspace():

            # Per the original tokenize_slice function, you need to add a
            # garbage token before the actual text you want to tokenize so that
            # the tokenizer does not add a beginning of sentence token.
            prefix = "a "

            # Tokenize the combined prefix and text
            wordpieces = self._tokenizer.tokenize(prefix + text_to_tokenize)

            # Go through each wordpiece in the tokenized wordpieces.
            for wordpiece in wordpieces:

                # Because we added the garbage prefix before tokenize, we need
                # to adjust the idx such that it accounts for this. Therefore we
                # subtract the length of the prefix from each token's idx.
                if wordpiece.idx is not None:
                    wordpiece.idx -= len(prefix)

            # We do not want the garbage token, so we return all but the first
            # token.
            return wordpieces[1:]
        else:

            # Do not need any sort of prefix, so just return all of the tokens.
            return self._tokenizer.tokenize(text_to_tokenize)

    def tokenize_str(self,
                     text: str) -> Iterable[Token]:
        """
        Helper method to tokenize a string.

        Adapted from the `transformer_squad.make_instances`

        Args:
            text: `str`
                The string to tokenize.

        Returns: `Iterable[Tokens]`
            The resulting tokens.

        """

        # We need to keep track of the current token index so that we can update
        # the results from self.tokenize_slice such that they reflect their
        # actual position in the string rather than their position in the slice
        # passed to tokenize_slice. Also used to construct the slice.
        token_index = 0

        # Create the output list (can be any iterable) that will store the
        # tokens we found.
        tokenized_str = []

        # Helper function to update the `idx` and add every wordpiece in the
        # `tokenized_slice` to the `tokenized_str`.
        def add_wordpieces(tokenized_slice: Iterable[Token]) -> None:
            for wordpiece in tokenized_slice:
                if wordpiece.idx is not None:
                    wordpiece.idx += token_index
                tokenized_str.append(wordpiece)

        # Iterate through every character and their respective index in the text
        # to create the slices to tokenize.
        for i, c in enumerate(text):
            i: int
            c: str

            # Check if the current character is a space. If it is, we tokenize
            # the slice of `text` from `token_index` to `i`.
            if c.isspace():
                add_wordpieces(self.tokenize_slice(text, token_index, i))
                token_index = i + 1

        # Add the end slice that is not collected by the for loop.
        add_wordpieces(self.tokenize_slice(text, token_index, len(text)))

        return tokenized_str

    @staticmethod
    def get_spans_from_text(text: str,
                            spans: List[Tuple[int, int]]) -> List[str]:
        """
        Helper function to get a span from a string

        Args:
            text: `str`
                The source string
            spans: `List[Tuple[int,int]]`
                List of start and end indices for spans.

                Assumes that the end index is inclusive. Therefore, for start
                index `i` and end index `j`, retrieves the span at `text[i:j+1]`.

        Returns: `List[str]`
            The extracted string from text.
        """
        return [text[start:end + 1] for start, end in spans]

    @overrides
    def text_to_instance(
            self,
            query: str,
            tokenized_query: List[Token],
            passage: str,
            tokenized_passage: List[Token],
            answers: List[str],
            token_answer_spans: Optional[Tuple[int, int]] = None,
            additional_metadata: Dict[str, Any] = None
    ) -> Instance:
        """
        TODO: Needs to Return
            question_with_context: Dict[str, Dict[str, torch.LongTensor]]
            context_span: torch.IntTensor
            cls_index: Optional[torch.LongTensor]
            answer_span: Optional[torch.IntTensor]
            metadata: Optional[List[Dict[str, Any]]]
        """
        fields = {}

        # Create the query field from the tokenized question and context. Use
        # `self._tokenizer.add_special_tokens` function to add the necessary
        # special tokens to the query.
        query_field = TextField(self._tokenizer.add_special_tokens(
            # The `add_special_tokens` function automatically adds in the
            # separation token to mark the separation between the two lists of
            # tokens. Therefore, we can create the query field WITH context
            # through passing them both as arguments.
            tokenized_query,
            tokenized_passage
        ), self._token_indexers)

        # Add the query field to the fields dict that will be outputted as an
        # instance. Do it here rather than assign above so that we can use
        # attributes from `query_field` rather than continuously indexing
        # `fields`.
        fields['question_with_context'] = query_field

        raise NotImplementedError(
            "'RecordTaskReader.text_to_instance' is not yet implemented"
        )
