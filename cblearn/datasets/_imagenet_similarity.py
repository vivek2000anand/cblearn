from pathlib import Path
import logging
import joblib
import os
from typing import Optional, Union

import numpy as np
from sklearn.datasets import _base
from sklearn.utils import check_random_state, Bunch

# The single files are downloaded individually instead of the complete OSF project archive
# (https://files.osf.io/v1/resources/7f96y/providers/osfstorage/?zip=).
# That archive is generated on-the-fly by the OSF server, is about 230MB large and
# has no stable checksum, which makes the download slow and unreliable.
OBSERVATIONS_V1 = _base.RemoteFileMetadata(
    filename='imagenet_similarity_obs-118.hdf5',
    url='https://osf.io/download/ej6sz/',
    checksum='d8a51e689ebd10c7c8a22eb9a52b16a7f18a8d205f4bce434bdaea5ad0d4fe61')
OBSERVATIONS_V2 = _base.RemoteFileMetadata(
    filename='imagenet_similarity_obs-195.hdf5',
    url='https://osf.io/download/x6dht/',
    checksum='2b4b22c1c5f425eea774e88fded389e188adc4424ba4eb9ca307a5ea732a3d76')
CATALOG = _base.RemoteFileMetadata(
    filename='imagenet_similarity_catalog.hdf5',
    url='https://osf.io/download/bf3e2/',
    checksum='920d3c577308b256fce2599726bb5eacd433a6250f710dc011048abf20cb85e4')

logger = logging.getLogger(__name__)
__doctest_requires__ = {'fetch_imagenet_similarity': ['h5py']}


def _import_h5py():
    """ Import the optional h5py dependency or raise an informative error. """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "This function needs the extra package 'h5py' but could not find it.\n"
            "The package can be installed with pip install h5py.\n"
            "On some platforms you might have to install hdf5 libraries separately.")
    return h5py


def _read_hdf5_arrays(path: os.PathLike) -> dict:
    """ Read all datasets of a HDF5 file into a dictionary of numpy arrays. """
    h5py = _import_h5py()
    with h5py.File(path, mode='r') as f:
        return {k: np.asarray(v[()]) for k, v in f.items()}


def fetch_imagenet_similarity(data_home: Optional[os.PathLike] = None, download_if_missing: bool = True,
                              shuffle: bool = True, random_state: Optional[np.random.RandomState] = None,
                              version: str = '0.1', return_data: bool = False) -> Union[Bunch, np.ndarray]:
    """ Load the imagenet similarity dataset (rank 2 from 8).

    ===================   =====================
    Trials    v0.1/v0.2        25,273 / 384,277
    Objects (Images)             1,000 / 50,000
    Classes                               1,000
    Query                         rank 2 from 8
    ===================   =====================

    See :ref:`imagenet_similarity_dataset` for a detailed description.

    .. Note :
        Loading dataset requires the package `h5py`_, which was not installed as an dependency of cblearn.

    .. _`h5py`: https://docs.h5py.org/en/stable/build.html

    >>> dataset = fetch_imagenet_similarity(shuffle=True, version='0.1')  # doctest: +REMOTE_DATA
    >>> dataset.class_label[[0, -1]].tolist()  # doctest: +REMOTE_DATA
    ['n01440764', 'n15075141']
    >>> dataset.n_select, dataset.is_ranked  # doctest: +REMOTE_DATA
    (2, True)
    >>> dataset.data.shape  # doctest: +REMOTE_DATA
    (25273, 9)

    Args:
        data_home : optional, default: None
            Specify another download and cache folder for the datasets. By default
            all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
        download_if_missing : optional, default=True
        shuffle: default = True
            Shuffle the order of triplet constraints.
        random_state: optional, default = None
            Initialization for shuffle random generator
        version: Version of the dataset.
            '0.1' contains one object per class,
            '0.2' 50 objects per class.
        return_triplets : boolean, default=False.
            If True, returns numpy array instead of a Bunch object.

    Returns:
        dataset : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data : ndarray, shape (n_query, 9)
                Each row corresponding a rank-2-of-8 query, entries are object indices.
                The first column is the reference, the second column is the most similar, and the
                third column is the second most similar object.
            rt_ms : ndarray, shape (n_query, )
                Reaction time in milliseconds.
            n_select : int
                Number of selected objects per trial.
            is_ranked : bool
                Whether the selection is ranked in similarity to the reference.
            session_id : (n_query,)
                Ids of the survey session for query recording.
            stimulus_id : (50.000,)
                Ids of the images.
            stimulus_filepath : (50.000,)
                Filepaths of images.
            class_id : (50.000,)
                ImageNet class assigned to each image.
            class_label : (1.000,)
                WordNet labels of the classes.
            DESCR : string
                Description of the dataset.
        data : numpy arrays (n_query, 9)
            Only present when `return_data=True`.

    Raises:
        IOError: If the data is not locally available, but download_if_missing=False
    """
    data_home = Path(_base.get_data_home(data_home=data_home))
    if not data_home.exists():
        data_home.mkdir()

    filepath = Path(_base._pkl_filepath(data_home, 'imagenet_similarity.pkz'))
    if not filepath.exists():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        _import_h5py()  # fail early, before downloading

        remote_files = (OBSERVATIONS_V1, OBSERVATIONS_V2, CATALOG)
        downloaded_paths = []
        for remote in remote_files:
            logger.info('Downloading imagenet similarity data from {} to {}'.format(remote.url, data_home))
            downloaded_paths.append(_base._fetch_remote(remote, dirname=data_home))

        data_v1, data_v2, catalog = [_read_hdf5_arrays(path) for path in downloaded_paths]

        joblib.dump((data_v1, data_v2, catalog), filepath, compress=6)
        for path in downloaded_paths:
            os.remove(path)
    else:
        (data_v1, data_v2, catalog) = joblib.load(filepath)

    if str(version) == '0.1':
        data = data_v1
    elif str(version) == '0.2':
        data = data_v2
    else:
        raise ValueError(f"Expects version '0.1' or '0.2', got '{version}'.")

    data.pop('trial_type')
    catalog['class_map_label'] = catalog['class_map_label'].astype(str)
    catalog['stimulus_filepath'] = catalog['stimulus_filepath'].astype(str)

    if shuffle:
        random_state = check_random_state(random_state)
        ix = random_state.permutation(len(data['stimulus_set']))
        data = {k: v[ix] for k, v in data.items()}

    if return_data:
        return data['stimulus_set']

    module_path = Path(__file__).parent
    with module_path.joinpath('descr', 'imagenet_similarity.rst').open() as rst_file:
        fdescr = rst_file.read()

    return Bunch(data=data['stimulus_set'],
                 rt_ms=data['rt_ms'],
                 n_select=int(np.unique(data['n_select'])[0]),
                 is_ranked=bool(np.unique(data['is_ranked'])[0]),
                 session_id=data['session_id'],
                 stimulus_id=catalog['stimulus_id'],
                 stimulus_filepath=catalog['stimulus_filepath'],
                 class_id=catalog['class_id'],
                 class_label=catalog['class_map_label'][1:],
                 DESCR=fdescr)
