import os
import errno
import hashlib
import requests
from tqdm import tqdm


def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.

    Parameters
    ----------
    path : str
        Path of the desired dir
    """

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

"""Download files with progress bar."""

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

"""Prepare PASCAL VOC datasets"""
import os
import shutil
import argparse
import tarfile

_TARGET_DIR = os.path.expanduser('~/.yolk/datasets/voc')

#####################################################################################
# Download and extract VOC datasets into ``path``

def download_voc(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://bit.ly/yolk_voc_train_val2007_tar',
        '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://bit.ly/yolk_voc_train_val2012_tar',
        '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://bit.ly/yolk_voc_test2012_tar',
        '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


#####################################################################################
# Download and extract the VOC augmented segmentation dataset into ``path``

def download_pascal(download_dir="~/VOCdevkit", overwrite=True, no_download=False):
    path = os.path.expanduser(download_dir)
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'VOC2007')) \
        or not os.path.isdir(os.path.join(path, 'VOC2012')):
        if no_download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you should not disable "--no-download" to grab it'.format(path)))
        else:
            download_voc(path, overwrite=overwrite)
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
            shutil.rmtree(os.path.join(path, 'VOCdevkit'))

    # make symlink
    makedirs(os.path.expanduser('~/.yolk/datasets'))
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    os.symlink(path, _TARGET_DIR)
    print("Downloaded!!!")


"""Prepare COCO datasets"""
import sys
import io
import shutil
import zipfile


def config():
    """
    Configurate download urls and hash values
        Parameters
        ----------
        Returns
        ----------
        dataset_path: 
            path where datasets are to be downloaded
        filenames: 
            file names to be downloaded
        urls: 
            download urls
        hashes: 
            MD5 hashes of files to be downloaded
    """
    dataset_path = os.path.join(os.path.expanduser('~'), '.yolk/datasets/coco')
    base_url = 'http://bit.ly/'
    postfix_urls = [
      	'yolk_coco_test2017_zip',
       	'yolk_coco_train2017_zip',
       	'yolk_coco_val2017_zip',
        'yolk_coco_annotation2017_zip'
    ]
    urls = [base_url + x for x in postfix_urls]
    image_filenames = [
       	'test2017.zip',
       	'train2017.zip',
       	'val2017.zip'
    ]
    annotation_filenames = [
        'annotations2017.zip'
    ]
    filenames = image_filenames + annotation_filenames
    return dataset_path, filenames, urls


"""
Download and Extract COCO Dataset
"""
def download_coco():
    dataset_path, filenames, urls = config()
 
    # clean and make directory before download datasets
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)
    else:
        os.makedirs(dataset_path)
    os.makedirs(os.path.join(dataset_path, 'images'))
    os.makedirs(os.path.join(dataset_path, 'annotations'))

    for filename, url in zip(filenames, urls):
        filepath = os.path.join(dataset_path, filename)
        with open(filepath, "wb") as file:
            print("Downloading %s" % filepath)
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None: # no content length header
                file.write(response.content)
            else:
                # show download progress bar
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    file.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                print()
        sub_dir = os.path.join(dataset_path, 'images')
        if 'annotations' in filename:
            sub_dir = os.path.join(dataset_path, 'annotations')
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(sub_dir)



def getHash(filepath):
    """
    Check download files with origin files
        Parameters
        ----------
        filepath: 
            path of the file
        Returns
        ----------
        hash value:
            md5 hash value of the file
    """
    blocksize=65536
    with open(filepath, "rb") as file:
        hasher = hashlib.md5()
        buf = file.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = file.read(blocksize)
        return hasher.hexdigest()


if __name__ == "__main__":
    pass