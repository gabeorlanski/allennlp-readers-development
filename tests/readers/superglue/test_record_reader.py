import pytest
from src.readers.superglue.record_reader import RecordTaskReader
from tests import FIXTURES_ROOT
from allennlp.data.tokenizers import WhitespaceTokenizer

"""
Tests for the ReCoRD reader from SuperGLUE
"""


class TestRecordReader:

    @pytest.fixture
    def whitespace_reader(self):
        reader = RecordTaskReader()
        # Set the tokenizer to whitespace tokenization for ease of use and
        # testing. Easier to test than using a transformer tokenizer.
        reader._tokenizer = WhitespaceTokenizer()
        yield reader

    @pytest.fixture
    def passage(self):
        return (
            "Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD) "
            "is a large-scale reading comprehension dataset which requires "
            "commonsense reasoning."
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

