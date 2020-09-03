import tempfile
from pathlib import Path

from double_jig_gen.io import get_abs_path_str, read_and_rstrip_file

NO_TRAILING_WHITESPACE_TMPFILE = tempfile.NamedTemporaryFile("w+")
TRAILING_WHITESPACE_TMPFILE = tempfile.NamedTemporaryFile("w+")


def setup_module():
    lines = ["example", "  lines", "    maybe", "trailing_whitespace"]
    NO_TRAILING_WHITESPACE_TMPFILE.write("\n".join(lines))
    tw_lines = ["example  ", "  lines  ", "    maybe", "trailing_whitespace"]
    TRAILING_WHITESPACE_TMPFILE.write("\n".join(tw_lines))


def teardown_module(module):
    NO_TRAILING_WHITESPACE_TMPFILE.close()
    TRAILING_WHITESPACE_TMPFILE.close()


def test_get_abs_path_str():
    assert get_abs_path_str(__file__) == __file__
    assert get_abs_path_str(Path(__file__)) == __file__


def test_read_and_strip_file():
    with open(NO_TRAILING_WHITESPACE_TMPFILE.name, "r") as filehandle:
        expected_str = filehandle.read()
        assert read_and_rstrip_file(TRAILING_WHITESPACE_TMPFILE.name) == expected_str
