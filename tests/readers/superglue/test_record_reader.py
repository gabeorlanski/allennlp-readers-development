import pytest
from src.readers.superglue.record_reader import RecordTaskReader
from tests import FIXTURES_ROOT
from allennlp.data.tokenizers import WhitespaceTokenizer
import re

"""
Tests for the ReCoRD reader from SuperGLUE
"""


class TestRecordReader:

    @pytest.fixture
    def whitespace_reader(self):
        # Set the tokenizer to whitespace tokenization for ease of use and
        # testing. Easier to test than using a transformer tokenizer.
        reader = RecordTaskReader(
            tokenizer=WhitespaceTokenizer(),
            length_limit=24,
            question_length_limit=8,
            stride=4
        )
        yield reader

    @pytest.fixture
    def passage(self):
        return (
            "Reading Comprehension with Commonsense Reasoning Dataset ( ReCoRD ) "
            "is a large-scale reading comprehension dataset which requires "
            "commonsense reasoning"
        )

    @pytest.fixture
    def record_name_passage(self, passage):
        """
        From the passage above, this is the snippet that contains the phrase
        "Reading Comprehension with Commonsense Reasoning Dataset". The returned
        object is a tuple with (start: int, end: int, text: str).
        """
        start = 0
        end = 56
        yield start, end, passage[start:end]

    @pytest.fixture
    def tokenized_passage(self, passage):
        tokenizer = WhitespaceTokenizer()
        return tokenizer.tokenize(passage)

    @pytest.fixture
    def answers(self):
        return [
            {'start': 58, 'end': 64, 'text': 'ReCoRD'},
            {'start': 128, 'end': 149, 'text': 'commonsense reasoning'},
            {'start': 256, 'end': 512, 'text': 'Should not exist'}
        ]

    @pytest.fixture
    def example_basic(self):

        return {
            "id"     : "dummy1",
            "source" : "ReCoRD docs",
            "passage": {
                'text'    : "ReCoRD contains 120,000+ queries from 70,000+ news articles. Each "
                            "query has been validated by crowdworkers. Unlike existing reading "
                            "comprehension datasets, ReCoRD contains a large portion of queries "
                            "requiring commonsense reasoning, thus presenting a good challenge "
                            "for future research to bridge the gap between human and machine "
                            "commonsense reading comprehension .",
                'entities': [
                    {'start': 0, 'end': 6},
                    {'start': 156, 'end': 162},
                    {'start': 250, 'end': 264},

                ]
            },
            'qas'    : [
                {
                    'id'     : 'dummyA1',
                    'query'  : '@placeholder is a dataset',
                    'answers': [
                        {'start': 0, 'end': 6, 'text': 'ReCoRD'},
                        {'start': 156, 'end': 162, 'text': 'ReCoRD'},
                    ]
                },
                {
                    'id'     : 'dummayA2',
                    'query'  : 'ReCoRD presents a @placeholder with the commonsense reading '
                               'comprehension task',
                    'answers': [
                        {'start': 250, 'end': 264, 'text': 'good challenge'},
                    ]

                }
            ]

        }

    #####################################################################
    # Unittests                                                         #
    #####################################################################
    def test_tokenize_slice_bos(self, whitespace_reader, passage, record_name_passage):
        """
        Test `tokenize_slice` with a string that is at the beginning of the
        text. This means that `start`=0.
        """
        result = list(whitespace_reader.tokenize_slice(
            passage, record_name_passage[0], record_name_passage[1]))

        assert len(result) == 6

        expected = ['Reading', 'Comprehension', 'with', 'Commonsense', 'Reasoning', 'Dataset']
        for i in range(len(result)):
            assert str(result[i]) == expected[i]

    def test_tokenize_slice_prefix(self, whitespace_reader, passage, record_name_passage):
        result = list(whitespace_reader.tokenize_slice(
            passage, record_name_passage[0] + 8, record_name_passage[1]))

        expected = ['Comprehension', 'with', 'Commonsense', 'Reasoning', 'Dataset']
        assert len(result) == len(expected)

        for i in range(len(result)):
            assert str(result[i]) == expected[i]

    def test_tokenize_str(self, whitespace_reader, record_name_passage):
        result = list(whitespace_reader.tokenize_str(record_name_passage[-1]))
        expected = ['Reading', 'Comprehension', 'with', 'Commonsense', 'Reasoning', 'Dataset']
        assert len(result) == len(expected)

        for i in range(len(result)):
            assert str(result[i]) == expected[i]

    def test_get_answer_offsets(self, whitespace_reader, tokenized_passage, answers):
        results = whitespace_reader.get_answer_offsets(tokenized_passage, answers)

        assert len(results) == 3
        assert results[0] == (7, 8)
        assert [t.text for t in tokenized_passage[results[0][0]:results[0][1]]] == [
            'ReCoRD']
        assert results[1] == (17, 19)
        assert [t.text for t in tokenized_passage[results[1][0]:results[1][1]]] == [
            'commonsense', 'reasoning']
        assert results[2] == (-1, -1)

    def test_get_instances_from_example(self, whitespace_reader, tokenized_passage, example_basic):
        # TODO: Make better
        result = list(whitespace_reader.get_instances_from_example(example_basic))

        assert len(result) == 2
        assert len(result[0]['question_with_context'].tokens) == whitespace_reader._length_limit
        assert '@placeholder' in [t.text for t in result[0]['question_with_context'].tokens]

        assert len(result[1]['question_with_context']) == whitespace_reader._length_limit
        assert '@placeholder' in [t.text for t in result[1]['question_with_context'].tokens]

    def test_get_instances_from_example_fields(self, whitespace_reader, tokenized_passage,
                                               example_basic):
        results = list(whitespace_reader.get_instances_from_example(example_basic))
        expected_keys = [
            "question_with_context",
            "context_span",
            # "cls_index",
            "answer_span",
            "metadata"
        ]
        for i in range(len(results)):
            assert len(results[i].fields) == len(expected_keys), f"results[{i}] has incorrect number of fields"
            for k in expected_keys:
                assert k in results[i].fields, f"results[{i}] is missing {k}"
