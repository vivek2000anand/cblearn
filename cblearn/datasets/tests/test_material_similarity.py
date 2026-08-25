import zipfile

import pytest

from cblearn.datasets._material_similarity import _archive_root


def _zip_with_names(path, names):
    with zipfile.ZipFile(path, 'w') as zf:
        for name in names:
            zf.writestr(name, 'x')
    return zipfile.ZipFile(path)


def test_archive_root_reads_the_single_top_level_directory(tmp_path):
    """ Test that the archive root is read from the zip instead of assumed.

    GitHub names that directory after the ref the archive was built from,
    so it changes whenever the pinned commit or branch changes.
    """
    zf = _zip_with_names(tmp_path / 'a.zip',
                         ['material-appearance-similarity-abc123/data/answers_processed_test.json',
                          'material-appearance-similarity-abc123/README.md'])
    assert _archive_root(zf) == 'material-appearance-similarity-abc123'


def test_archive_root_rejects_ambiguous_archives(tmp_path):
    zf = _zip_with_names(tmp_path / 'b.zip', ['one/file.txt', 'two/file.txt'])
    with pytest.raises(IOError, match="single top level directory"):
        _archive_root(zf)
