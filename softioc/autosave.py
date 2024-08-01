import atexit
import numpy
import sys
import threading
import traceback
import time
import yaml

from datetime import datetime
from os import rename
from pathlib import Path
from shutil import copy2

SAV_SUFFIX = "softsav"
SAVB_SUFFIX = "softsavB"
DEFAULT_SAVE_PERIOD = 30.0
DEFAULT_FILENAME = "auto"


def _ndarray_representer(dumper, array):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", array.tolist(), flow_style=True
    )

def configure_directory(directory, enabled=True):
    __Autosave.directory = Path(directory)
    __Autosave.enabled = enabled

def configure_file(filename=DEFAULT_FILENAME, save_period=30, timestamped_backups=True, enabled=True):
    __Autosave.files[filename] = __AutosaveFile(filename, save_period, timestamped_backups, enabled)

class __AutosaveFile:
    def __init__(self, filename, save_period, timestamped_backups, enabled):
        self.filename = filename
        self.save_period = save_period
        self.timestamped_backups = timestamped_backups
        self.enabled = enabled
        self.last_saved_time = datetime.now()
        self.pvs = {}
        self.last_state = {}


def start_autosave_thread():
    autosaver = __Autosave()
    worker = threading.Thread(
        target=autosaver.loop,
    )
    worker.daemon = True
    worker.start()
    atexit.register(_shutdown_autosave_thread, autosaver, worker)


def _shutdown_autosave_thread(autosaver, worker):
    autosaver.stop()
    worker.join()


def add_pv_to_autosave(pv, name, save_val, save_fields, filename):
    """Configures a PV for autosave

    Args:
        pv: a PV object inheriting ProcessDeviceSupportCore
        name: the name of the PV which is used to generate the key
            by which the PV value is saved to and loaded from a backup,
            this is typically the signal name without the device prefix
        save_val: a boolean that tracks whether to save the VAL field
            in an autosave backup
        save_fields: a list of string names of fields associated with the pv
            to be saved to and loaded from a backup
    """
    if not filename:
        filename = DEFAULT_FILENAME
    if filename not in __Autosave.files:
        raise Exception("Some kind of error here TODO")
    file = __Autosave.files[filename]
    if save_val:
        file.pvs[name] = _AutosavePV(pv)
    for field in save_fields:
        file.pvs[name] = __AutosaveFile(pv, field)


def load_autosave():
    __Autosave._load()


class _AutosavePV:
    def __init__(self, pv, field=None):
        if not field or field == "VAL":
            self.get = pv.get
            self.set = pv.set
        else:
            self.get = lambda: pv.get_field(field)
            self.set = lambda val: setattr(pv, field, val)


class __Autosave:
    _pvs = {}
    _last_saved_time = datetime.now()
    _stop_event = threading.Event()
    save_period = DEFAULT_SAVE_PERIOD
    directory = None
    enabled = False
    files = {}

    def __init__(self):
        if not self.enabled:
            return
        yaml.add_representer(
            numpy.ndarray, _ndarray_representer, Dumper=yaml.Dumper
        )
        if not self.directory:
            raise RuntimeError(
                "Autosave directory is not known, call "
                "autosave.configure() with keyword argument "
                "directory"
            )
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"{self.directory} is not a valid autosave directory"
            )
    
    @classmethod
    def _get_sav_file(cls, file):
        return cls.directory / f"{file.filename}.{SAV_SUFFIX}"

    @classmethod
    def _get_tmp_file(cls, file):
        return cls.directory / f"{file.filename}.{SAVB_SUFFIX}"

    @classmethod
    def _load(cls):
        for file in [f for f in cls.files.values() if f.enabled]:
            sav_path = cls._get_sav_file(file)
            if not sav_path.is_file():
                print(sav_path, "not a file")
                continue
            if file.timestamped_backups:
                bu_file = sav_path.parent / f"{sav_path.name}{file.last_saved_time.strftime('_%y%m%d-%H%M%S')}"
            else:
                bu_file = cls.directory / f"{file.filename}.bu" 
            copy2(sav_path, bu_file)

            try:
                with open(sav_path, "r") as f:
                    state = yaml.full_load(f)
                    cls._set_pvs_from_state(file, state)

            except Exception:
                traceback.print_exc()
    
    @classmethod
    def _set_pvs_from_state(cls, file, state):
        for pv_field, value in state.items():
            try:
                pv = file.pvs[pv_field]
                pv.set(value)
            except Exception:
                print(
                    f"Exception setting {pv_field} to {value}",
                    file=sys.stderr,
                )
                traceback.print_exc()

    def _get_state(self, file):
        state = {}
        for pv_field, pv in file.pvs.items():
            try:
                state[pv_field] = pv.get()
            except Exception:
                print(f"Exception getting {pv_field}", file=sys.stderr)
                traceback.print_exc()
        return state

    def _state_changed(self, state, file):
        return state.keys() != file.last_state.keys() or any(
            not numpy.array_equal(state[key], file.last_state[key])
            for key in state
        )
        # checks equality for builtins and numpy arrays

    def _save(self, file, now, force=False):
        print(file, now)
        if (now - file.last_saved_time).seconds < file.save_period and not force:
            return
        state = self._get_state(file)
        if not self._state_changed(state, file):
            return
        # else do the saving
        # TODO: __AutosaveFile needs to hold a last_saved_state dict.
        sav_path = self._get_sav_file(file)
        tmp_path = self._get_tmp_file(file)
        with open(tmp_path, "w") as tmp:
            yaml.dump(state, tmp, indent=4)
        rename(tmp_path, sav_path)
        file.last_state = state
        file.last_saved_time = now  # or should we call datetime.now() again?

    def _save_all(self, now):
        for file in self.files.values():
            self._save(file, now, force=True)

    def loop(self):  # should we check if asyncio is available and if so, use it?? it would be better probably that calling time
        # over and over...
        if not self.enabled or not self.files:
            return
        while True:
            now = datetime.now()  # is there a better way to do this???
            try:
                for file in self.files.values():
                    if self._stop_event.is_set():
                        # stop requested, save every file then exit
                        self._save_all(now)
                    else:
                        self._save(file, now)
            except Exception:
                traceback.print_exc()

    def stop(self):
        self._stop_event.set()