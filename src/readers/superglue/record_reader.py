"""
Dataset reader for SuperGLUE's Reading Comprehension with Commonsense Reasoning task (Zhang Et
al. 2018).

Reader Implemented by Gabriel Orlanski
"""

from typing import Dict, List, Optional, Iterable, Union, Tuple, Any
from pathlib import Path

from overrides import overrides
from src.util.log_util import getBothLoggers
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
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
                 transformer_model_name: str = "bert-base-cased",
                 length_limit: int = 384,
                 question_length_limit: int = 64,
                 stride: int = 128,
                 raise_errors: bool = False,
                 tokenizer_kwargs: Dict[str, Any] = None,
                 max_instances: int=None,
                 **kwargs) -> None:
        """
        Initialize the RecordTaskReader.
        """
        super(RecordTaskReader, self).__init__(
            manual_distributed_sharding=True, **kwargs
        )

        # Save the values passed to __init__ to protected attributes
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name,
            add_special_tokens=False,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                transformer_model_name, tokenizer_kwargs=tokenizer_kwargs
            )
        }
        self._length_limit = length_limit
        self._query_len_limit = question_length_limit
        self._stride = stride
        self._raise_errors = raise_errors
        self._cls_token = '@placeholder'
        self._max_instances = max_instances

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

            passages_yielded += instance_count

            # Check to see if we are over the max_instances to yield.
            if self._max_instances and passages_yielded > self._max_instances:
                logger.info(f"Passed max instances")
                break


        # Log pertinent information.
        if passages_yielded:
            logger.info(f"{examples_multiple_instance_count}/{passages_yielded} "
                        f"({examples_multiple_instance_count / passages_yielded * 100:.2f}%) "
                        f"examples had more than one instance")
        else:
            logger.warning(f"Could not find any instances in '{file_path}'")

    def get_instances_from_example(self,
                                   example: Dict,
                                   always_add_answer_span: bool = False) -> Iterable[Instance]:
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
        tokenized_passage: List[Token] = self.tokenize_str(passage_text)

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

            # Calculate where the context needs to start and how many tokens we have
            # for it. This is due to the limit on the number of tokens that a
            # transformer can use because they have quadratic memory usage. But if
            # you are reading this code, you probably know that.
            space_for_context = (
                    self._length_limit
                    - len(list(tokenized_query))

                    # Used getattr so I can test without having to load a
                    # transformer model.
                    - len(getattr(self._tokenizer, 'sequence_pair_start_tokens', []))
                    - len(getattr(self._tokenizer, 'sequence_pair_mid_tokens', []))
                    - len(getattr(self._tokenizer, 'sequence_pair_end_tokens', []))
            )

            # Check if answers exist for this query. We assume that there are no
            # answers for this query, and set the start and end index for the
            # answer span to -1.
            answers = query.get('answers', [])
            if not answers:
                logger.warning(f"Skipping {query['id']}, no answers")
                continue

            # Get the token offsets for the answers for this current passage.
            answer_token_start, answer_token_end = (-1, -1)
            for offsets in self.get_answer_offsets(tokenized_passage, answers):
                if offsets != (-1, -1):
                    answer_token_start, answer_token_end = offsets
                    break

            # Go through the context and find the window that has the answer in it.
            stride_start = 0

            while True:
                tokenized_context_window = tokenized_passage[stride_start:]
                tokenized_context_window = tokenized_context_window[:space_for_context]

                # Get the token offsets w.r.t the current window.
                window_token_answer_span = (
                    answer_token_start - stride_start,
                    answer_token_end - stride_start,
                )
                if any(i < 0 or i >= len(tokenized_context_window) for i in
                       window_token_answer_span):
                    # The answer is not contained in the window.
                    window_token_answer_span = None

                if (
                        # not self.skip_impossible_questions
                        window_token_answer_span is not None
                ):
                    # The answer WAS found in the context window, and thus we
                    # can make an instance for the answer.
                    instance = self.text_to_instance(
                        query['query'],
                        tokenized_query,
                        passage_text,
                        tokenized_context_window,
                        answers=answers,
                        token_answer_span=window_token_answer_span,
                        additional_metadata=additional_metadata,
                        always_add_answer_span=always_add_answer_span,
                    )
                    yield instance

                stride_start += space_for_context

                # If we have reached the end of the passage, stop.
                if stride_start >= len(tokenized_passage):
                    break

                # I am not sure what this does...but it is here?
                stride_start -= self._stride

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
                     text: str) -> List[Token]:
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
            token_answer_span: Optional[Tuple[int, int]] = None,
            additional_metadata: Optional[Dict[str, Any]] = None,
            always_add_answer_span: Optional[bool] = False,
    ) -> Instance:
        """
        A lot of this comes directly from the `transformer_squad.text_to_instance`
        TODO: Improve docs
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

        # Calculate the index that marks the start of the context.
        start_of_context = (
                + len(tokenized_query)
                # Used getattr so I can test without having to load a
                # transformer model.
                - len(getattr(self._tokenizer, 'sequence_pair_start_tokens', []))
                - len(getattr(self._tokenizer, 'sequence_pair_mid_tokens', []))
        )

        # make the answer span
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]

            fields["answer_span"] = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                query_field,
            )
        # make the context span, i.e., the span of text from which possible
        # answers should be drawn
        fields["context_span"] = SpanField(
            start_of_context, start_of_context + len(tokenized_passage) - 1, query_field
        )

        # make the metadata
        metadata = {
            "question"       : query,
            "question_tokens": tokenized_query,
            "context"        : passage,
            "context_tokens" : tokenized_passage,
            "answers"        : answers or [],
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    # TODO: Optimize because python while loops are SLOW
    def get_answer_offsets(self,
                           tokenized_passage: List[Token],
                           answers: List[Dict]) -> List[Tuple[int, int]]:
        out = []

        # Answers are ordered by which shows up first in the text (I hope).
        # Therefore, we keep track of what token we are on to avoid
        # checking the same tokens more than once.
        current_token_index = 0
        current_answer = 0

        while current_answer < len(answers):
            answer: Dict = answers[current_answer]

            # We tokenize the actual answer text. This is then used
            # immediately to help locate its start and end indices in
            # the tokenized passage. While ReCoRD does provide a start
            # and end index, it is only for the string itself rather
            # than the tokens
            tokenized_answer = self.tokenize_str(answer['text'])

            # Set a separate tracker for the current index.
            token_index = current_token_index

            # Rather than having this in the main check, have it as a sub loop
            # so that we can still add the -1 indices if the answers could not
            # be found. Does produce an issue where one malformed answer causes
            # the rest to be skipped. That is where `token_index` comes in. W/ a
            # separate tracker, we can pass on updating `current_token_index` if
            # we could not find the answer. This allows us to combat malformed
            # answers.
            found_answer = False
            while token_index < len(tokenized_passage):

                if tokenized_passage[token_index] == tokenized_answer[0]:
                    valid_answer = True
                    for i in range(1, len(tokenized_answer)):

                        # If there are no more tokens left in the passage,
                        # or the answer and the current token no longer match.
                        if (
                                token_index + i > len(tokenized_passage)
                                or tokenized_passage[token_index + i] != tokenized_answer[i]
                        ):
                            valid_answer = False
                            break
                    if valid_answer:
                        token_index += len(tokenized_answer)
                        found_answer = True
                        break
                token_index += 1

            # We found an answer, add it to the output.
            if found_answer:
                current_token_index = token_index + 1
                out.append((token_index - len(tokenized_answer), token_index))
            else:
                out.append((-1, -1))

            current_answer += 1

        return out

    def _find_cls_index(self, tokens: List[Token]) -> int:
        """
        From transformer_squad
        Args:
            self:
            tokens:

        Returns:

        """
        return next(i for i, t in enumerate(tokens) if t.text == self._cls_token)
