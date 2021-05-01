import pathlib

PROJECT_ROOT = (pathlib.Path(__file__).parent.joinpath("..")).resolve()  # pylint: disable=no-member
TESTS_ROOT = PROJECT_ROOT.joinpath("tests")
FIXTURES_ROOT = TESTS_ROOT.joinpath("fixtures")
