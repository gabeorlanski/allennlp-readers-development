import pytest
from src.readers.superglue.record_reader import RecordTaskReader
from allennlp.data.tokenizers import WhitespaceTokenizer
import re

"""
Tests for the ReCoRD reader from SuperGLUE
"""


# TODO: Add full integration tests

class TestRecordReader:

    @pytest.fixture
    def reader(self):
        yield RecordTaskReader(
            length_limit=256
        )

    @pytest.fixture
    def whitespace_reader(self, reader):
        # Set the tokenizer to whitespace tokenization for ease of use and
        # testing. Easier to test than using a transformer tokenizer.
        reader._tokenizer = WhitespaceTokenizer()
        reader._length_limit = 24
        reader._query_len_limit = 8
        reader._stride = 4
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

    @pytest.fixture
    def curiosity_example(self):
        """
        Bug where most examples did not return any instances, so doing
        regression testing on this real example that did not return anything.
        """
        return {
            'id'     : 'd978b083f3f97a2ab09771c72398cfbac094f818',
            'source' : 'Daily mail',
            'passage': {
                'text'    : "By Sarah Griffiths PUBLISHED: 12:30 EST, 10 July 2013 | UPDATED: "
                            "12:37 EST, 10 July 2013 Nasa's next Mars rover has been given a "
                            "mission to find signs of past life and to collect and store rock "
                            "from the the red planet that will one day be sent back to Earth. It "
                            "will demonstrate technology for a human exploration of the planet "
                            "and look for signs of life. The space agency has revealed what the "
                            "rover, known as Mars 2020, will look like. Scroll down for video... "
                            "Nasa's next Mars rover (plans pictured) has been given a mission to "
                            "find signs of past life and to collect and store rock from the the "
                            "red planet that will one day be sent back to Earth. Mars 2020 will "
                            "also demonstrate technology for a human exploration of the "
                            "planet\n@highlight\nMars 2020 will collect up to 31 rock and soil "
                            "samples from the red planet and will look for signs of "
                            "extraterrestrial life\n@highlight\nThe new rover will use the same "
                            "landing system as Curiosity and share its frame, which has saved "
                            "Nasa $1 billion\n@highlight\nThe mission will bring the sapec agency "
                            "a step closer to meeting President Obama's challenge to send humans "
                            "to Mars in the next decade",
                'entities': [
                    {'start': 3, 'end': 17}, {'start': 89, 'end': 92}, {'start': 101, 'end': 104},
                    {'start': 252, 'end': 256}, {'start': 411, 'end': 414},
                    {'start': 463, 'end': 466}, {'start': 475, 'end': 478},
                    {'start': 643, 'end': 647}, {'start': 650, 'end': 653},
                    {'start': 742, 'end': 745}, {'start': 926, 'end': 934},
                    {'start': 973, 'end': 976}, {'start': 1075, 'end': 1079},
                    {'start': 1111, 'end': 1114}]
            },
            'qas'    : [
                {
                    'id'     :
                        'd978b083f3f97a2ab09771c72398cfbac094f818'
                        '-04b6e904611f0d706521db167a05a11bf693e40e-61',
                    'query'  : 'The 2020 mission plans on building on the accomplishments of '
                               '@placeholder and other Mars missions.',
                    'answers': [
                        {'start': 926, 'end': 934, 'text': 'Curiosity'}]
                }]
        }

    @pytest.fixture
    def skyfall_example(self):
        """
        Another example that was not returning instances
        """
        return {
            'id': '6f1ca8baf24bf9e5fc8e33b4b3b04bd54370b25f', 'source': 'Daily mail', 'passage': {
            'text'    : "They're both famous singers who have lent their powerful voices to James "
                        "Bond films. And it seems the Oscars' stage wasn't big enough to "
                        "accommodate larger-and-life divas Adele and Dame Shirley Bassey, "
                        "at least at the same time. Instead of the two songstresses dueting or "
                        "sharing the stage, each performed her theme song separately during "
                        "Sunday night's ceremony. Scroll down for video Battle of the divas: "
                        "Adele and Dame Shirley Bassey separately sang James Bond theme songs "
                        "during Sunday night's Oscar ceremony Shirley performed first, "
                        "singing Goldfinger nearly 50 years since she first recorded the song for "
                        "the 1964 Bond film of the same name.\n@highlight\nAdele awarded Oscar "
                        "for Best Original Score for Skyfall",
            'entities': [{'start': 67, 'end': 76}, {'start': 102, 'end': 107},
                         {'start': 171, 'end': 175}, {'start': 181, 'end': 199},
                         {'start': 407, 'end': 411}, {'start': 417, 'end': 435},
                         {'start': 453, 'end': 462}, {'start': 498, 'end': 502},
                         {'start': 513, 'end': 519}, {'start': 546, 'end': 555},
                         {'start': 620, 'end': 623}, {'start': 659, 'end': 663},
                         {'start': 673, 'end': 701}, {'start': 707, 'end': 713}]
        }, 'qas': [{
                       'id'     : '6f1ca8baf24bf9e5fc8e33b4b3b04bd54370b25f-98823006424cc595642b5ae5fa1b533bbd215a56-105',
                       'query'  : 'The full works: Adele was accompanied by an orchestra, choir and light display during her performance of @placeholder',
                       'answers': [{'start': 707, 'end': 713, 'text': 'Skyfall'}]
                   }]
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
            assert len(results[i].fields) == len(
                expected_keys), f"results[{i}] has incorrect number of fields"
            for k in expected_keys:
                assert k in results[i].fields, f"results[{i}] is missing {k}"

    #####################################################################
    # Regression Test                                                   #
    #####################################################################

    def test_get_instances_from_example_curiosity(self, reader, curiosity_example):
        tokenized_answer = ' '.join(map(str, reader.tokenize_str('Curiosity')))
        results = list(reader.get_instances_from_example(curiosity_example))
        assert len(results) == 2
        assert tokenized_answer in ' '.join(map(str, results[0]['question_with_context'].tokens))
        assert tokenized_answer in ' '.join(map(str, results[1]['question_with_context'].tokens))

        # TODO: Make this its own test.
        # Kind of forced this extra test in here because I added it while
        # solving this bug, so just left it instead of creating another
        # unittest.
        reader._one_instance_per_query = True
        results = list(reader.get_instances_from_example(curiosity_example))
        assert len(results) == 1
        assert tokenized_answer in ' '.join(map(str, results[0]['question_with_context'].tokens))

    def test_get_answer_offsets_curiosity(self, reader, curiosity_example):
        offsets = reader.get_answer_offsets(reader.tokenize_str(
            curiosity_example['passage']['text']),
            curiosity_example['qas'][0]['answers']
        )
        assert len(offsets) == 1
        assert offsets[0] == (217, 221)
