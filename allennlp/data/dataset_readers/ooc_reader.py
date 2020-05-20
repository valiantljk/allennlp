import logging

import jsonpickle
import sys
from typing import Optional

import dill

from torch.utils.data import Dataset

from allennlp.data.instance import Instance
from allennlp.data.fields import Field, SequenceField, SequenceLabelField, TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


_pickle_exceptions = [Instance, TextField, SequenceField, SequenceLabelField, Field]  # TODO: Add more exceptions
_pickle_exception_ids = [id(o) for o in _pickle_exceptions]


class _Pickler(dill.Pickler):
    def persistent_id(self, obj):
        try:
            return _pickle_exception_ids.index(id(obj))
        except ValueError:
            return None


class _Unpickler(dill.Unpickler):
    def persistent_load(self, pid):
        return _pickle_exceptions[pid]


class OOCDataset(Dataset):
    """
    An implementation of Dataset that can be fed lazily on the input side but pretends to be a
    map-style dataset on the output side. Calls that can not yet be fulfilled because the data
    is not available yet block. Instances are cached to disk so you can use this with large
    datasets. The name is short for "Out Of Core Dataset".
    """
    def __init__(
        self,
        instance_serializer = jsonpickle.dumps,
        instance_deserializer = jsonpickle.loads,
        vocab: Vocabulary = None
    ):
        self.instance_serializer = instance_serializer
        self.instance_deserializer = instance_deserializer
        self.vocab = vocab

        import threading
        self.len: Optional[int] = None
        self.lock = threading.Lock()

        import tempfile
        self.dbdir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}-")
        logger.info("Caching dataset in %s", self.dbdir)
        import os
        import lmdb
        self.lmdb_env = lmdb.open(
            self.dbdir,
            subdir=True,
            map_size=1024*1024*1024*1024,
            max_readers=os.cpu_count() * 2,
            max_dbs=0,
            # The rest of these settings are optimizations. Since we don't care about
            # corruption in the case of a system crash, we turn them all on.
            metasync=False,
            sync=False,
            writemap=True,
            meminit=False,
            map_async=True)

        self.write_event = threading.Event()

        self.error: Optional[BaseException] = None

    def __del__(self):
        # TODO: Figure out if this gets called on Ctrl-C
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

        if self.dbdir is not None:
            import shutil
            shutil.rmtree(self.dbdir, ignore_errors=True)
            self.dbdir = None

    def __setitem__(self, key, value):
        self._check_error()
        key = key.to_bytes(4, byteorder=sys.byteorder)
        value = self.instance_serializer(value)
        import io
        with io.BytesIO() as buffer:
            pickler = _Pickler(buffer)
            pickler.dump(value)
            with self.lmdb_env.begin(write=True) as write_txn:
                write_txn.put(key, buffer.getbuffer(), overwrite=True)
        self.write_event.set()

    def __getitem__(self, key: int):
        key = key.to_bytes(4, byteorder=sys.byteorder)
        import io
        while True:
            self._check_error()
            with self.lmdb_env.begin(buffers=True) as read_txn:
                r = read_txn.get(key, default=None)
                if r is not None:
                    unpickler = _Unpickler(io.BytesIO(r))
                    r = unpickler.load()
                    break
            flag = self.write_event.wait(30)
            if flag:
                self.write_event.clear()
            else:
                logger.info("Waited more than 30 seconds for a write to OOCDataset. Continuing to wait.")
        return self.instance_deserializer(r)

    def set_length(self, len: int) -> None:
        with self.lock:
            if self.len is not None:
                raise ValueError("Length already set.")
            self.len = len

        # We use the same event for writes and for length so that we get the same warning when we haven't
        # seen any activity for 30 seconds.
        self.write_event.set()

    def __len__(self) -> int:
        while self.len is None:
            self._check_error()
            flag = self.write_event.wait()
            if flag:
                self.write_event.clear()
            else:
                logger.info("Waited more than 30 seconds for a write to OOCDataset. Continuing to wait.")
        return self.len

    def set_error(self, error: BaseException) -> None:
        if error is None:
            raise ValueError("You cannot unset the error.")
        with self.lock:
            if self.error is not None:
                raise ValueError("Error already set")
            self.error = error

    def _check_error(self):
        # If setting self.error is atomic, and it never gets reset, we don't need a lock here.
        if self.error is not None:
            raise self.error

    def index_with(self, vocab: Vocabulary) -> None:
        self.vocab = vocab


class OOCDatasetReader(DatasetReader):
    def __init__(self, length: Optional[int] = None):
        super(OOCDatasetReader, self).__init__(
            lazy=False,
            cache_directory=None,
            manual_distributed_sharding=True,
            max_instances=length)
        self.length = length

    def read(self, file_path: str) -> Dataset:
        r = OOCDataset()
        if self.length is not None:
            r.set_length(self.length)

        def fill_ooc_dataset():
            try:
                instances = self._read(file_path)
                if self.length is not None:
                    import itertools
                    instances = itertools.islice(instances, self.length)
                idx = -1
                for idx, instance in enumerate(instances):
                    r[idx] = instance
                instances_written = idx + 1
                if self.length is None:
                    r.set_length(instances_written)
                elif self.length > instances_written:
                    raise ValueError(
                        "The reader promised %d instances, but we have only %d",
                        self.length,
                        instances_written)
            except BaseException as e:
                r.set_error(e)

        import threading
        reading_thread = threading.Thread(
            target=fill_ooc_dataset,
            name=f"{self.__class__.__name__}-reader")
        reading_thread.daemon = True
        reading_thread.start()

        return r
