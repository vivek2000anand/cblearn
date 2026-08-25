import csv
from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch

# The single files are downloaded individually instead of the complete OSF project archive
# (https://files.osf.io/v1/resources/z2784/providers/osfstorage/?zip=).
# That archive is generated on-the-fly by the OSF server, is more than 180MB large and
# has no stable checksum, which makes the download slow and unreliable.
TRIPLETS = _base.RemoteFileMetadata(
    filename='things_data1854_batch5_test10.txt',
    url='https://osf.io/download/sd3f7/',
    checksum='789fb927c87538d29e0cf294793280cf5a77d45dc5fdd87cf8c726993397adb8')
OBJECT_NAMES = _base.RemoteFileMetadata(
    filename='things_items1854names.tsv',
    url='https://osf.io/download/sz2ux/',
    checksum='03dbe77972f9466ff8baa0da604d9f5104f677bf77cc602faebfa03a290fb51c')

logger = logging.getLogger(__name__)


def fetch_things_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                            shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                            return_data: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the things similarity dataset (odd-one-out).

    ===================   =====================
    Trials                              146,012
    Objects (Things)                      1,854
    Query                 3 images, odd one out
    ===================   =====================

    See :ref:`things_similarity_dataset` for a detailed description.

    >>> dataset = fetch_things_similarity(shuffle=True)  # doctest: +REMOTE_DATA
    >>> dataset.word[[0, -1]].tolist()  # doctest: +REMOTE_DATA
    ['aardvark', 'zucchini']
    >>> dataset.data.shape  # doctest: +REMOTE_DATA
    (146012, 3)

    Args:
        data_home : optional, default: None
            Specify another download and cache folder for the datasets. By default
            all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
        download_if_missing : optional, default=True
        shuffle: default = True
            Shuffle the order of triplet constraints.
        random_state: optional, default = None
            Initialization for shuffle random generator
        return_triplets : boolean, default=False.
            If True, returns numpy array instead of a Bunch object.

    Returns:
        dataset : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data : ndarray, shape (n_query, 3)
                Each row corresponding a odd-one-out query, entries are object indices.
                The first column is the selected odd-one.
            word : (n_objects,)
                Single word associated with the thing objects.
            synset : (n_objects,)
                Wordnet Synset associated with the thing objects.
            wordnet_id : (n_objects,)
                Wordnet Id associated with the thing objects.
            thing_id : (n_objects,)
                Unique Id string associated with the thing objects.
            DESCR : string
                Description of the dataset.
        data : numpy arrays (n_query, 3)
            Only present when `return_data=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'things_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        remote_files = (TRIPLETS, OBJECT_NAMES)
        downloaded_paths = []
        for remote in remote_files:
            logger.info('Downloading things similarity data from {} to {}'.format(remote.url, data_home))
            downloaded_paths.append(_base._fetch_remote(remote, dirname=data_home))
        triplets_path, names_path = downloaded_paths

        data = np.loadtxt(triplets_path, delimiter=' ')
        with open(names_path, 'r', newline='', encoding='utf-8') as f:
            objects = np.array(list(csv.reader(f, dialect='excel-tab'))[1:]).T

        joblib.dump((data, objects), filepath, compress=6)
        for path in downloaded_paths:
            os.remove(path)
    else:
        (data, objects) = joblib.load(filepath)

    if shuffle:
        random_state = check_random_state(random_state)
        data = random_state.permutation(data)

    if return_data:
        return data

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'things_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=data,
                 word=objects[0],
                 synset=objects[1],
                 wordnet_id=objects[2],
                 thing_id=objects[5],
                 DESCR=fdescr)
