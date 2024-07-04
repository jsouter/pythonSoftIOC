import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

from softioc.device_core import LookupRecord, LookupRecordList

SAV_SUFFIX = "softsav"
SAVB_SUFFIX = "softsavB"


def configure(directory=None, save_period=None, device=None):
    Autosave.save_period = save_period or Autosave.save_period
    if device is None:
        if Autosave.device_name is None:
            from .builder import GetRecordNames

            Autosave.device_name = GetRecordNames().prefix[0]
    else:
        Autosave.device_name = device
    if directory is None and Autosave.directory is None:
        raise RuntimeError(
            "Autosave directory is not known, call "
            "autosave.configure() with keyword argument "
            "directory."
        )
    else:
        Autosave.directory = Path(directory)

def set_autosave(pv, value=True):
    if value:
        Autosave.add_pv(pv)
    else:
        Autosave.remove_pv(pv)

def load_req_file(file, override=False):
    with open(file, "r") as f:
        pv_names = [name.strip() for name in f.readlines()]
        if not override:
            for pv_name in pv_names:
                pv = LookupRecord(pv_name)
                set_autosave(pv, True)
        else:  # explicitly set autosave for False if pv not in file
            for pv_name, pv in LookupRecordList():
                set_autosave(pv, pv_name in pv_names)


class Autosave:
    _instance = None
    _pvs = {}
    save_period = 30.0
    device_name = None
    directory = None
    enabled = True
    backup_on_restart = True

    def __init__(self):
        if not self.directory:
            raise RuntimeError(
                "Autosave directory is not known, call "
                "autosave.configure() with keyword argument "
                "directory."
            )
        if not self.device_name:
            raise RuntimeError(
                "Device name is not known to autosave thread, "
                "call autosave.configure() with device keyword argument"
            )
        self._last_saved_time = datetime.now()
        if not self.directory.is_dir():
            raise RuntimeError(
                f"{self.directory} is not a valid autosave directory"
            )
        if self.backup_on_restart:
            self.backup_sav_file()
        self.get_autosave_pvs()
        self._state = {}
        self._last_saved_state = {}
        self._stop_event = threading.Event()
        if self.enabled:
            self.load()  # load at startup if enabled

    def get_autosave_pvs(self):
        self._pvs = {name: pv for name, pv in LookupRecordList() if pv.autosave}

    def _change_directory(self, directory: str):
        dir_path = Path(directory)
        if dir_path.is_dir():
            self.directory = dir_path
        else:
            raise ValueError(f"{directory} is not a valid autosave directory")

    def backup_sav_file(self):
        sav_path = self._get_current_sav_path()
        if sav_path.is_file():
            shutil.copy2(sav_path, self._get_timestamped_backup_sav_path())

    @classmethod
    def add_pv(cls, pv):
        pv.set_autosave(True)
        cls._pvs[pv._name] = pv

    @classmethod
    def remove_pv(cls, pv):
        pv.set_autosave(False)
        cls._pvs.pop(pv._name, None)

    def _get_timestamped_backup_sav_path(self):
        sav_path = self._get_current_sav_path()
        return sav_path.parent / (
            sav_path.name + self._last_saved_time.strftime("_%y%m%d-%H%M%S")
        )

    def _get_backup_save_path(self):
        return self.directory / f"{self.device_name}.{SAVB_SUFFIX}"

    def _get_current_sav_path(self):
        return self.directory / f"{self.device_name}.{SAV_SUFFIX}"

    def _save(self, state):
        try:
            for path in [
                self._get_current_sav_path(),
                self._get_backup_save_path()
            ]:
                with open(path, "w") as f:
                    json.dump(state, f, indent=4)
            self._last_saved_state = state.copy()  # do we need to copy?
            self._last_saved_time = datetime.now()
        except Exception as e:
            print(f"Could not save state to file: {e}")

    def save(self):
        if not self.enabled:
            print("Not saving to file as autosave adapter disabled")
            return
        state = {name: pv.get() for name, pv in self._pvs.items()}
        if state != self._last_saved_state:
            self._save(state)

    def load(self, path=None):
        if not self.enabled:
            print("Not loading from file as autosave adapter disabled")
            return
        sav_path = path or self._get_current_sav_path()
        if not sav_path or not sav_path.is_file():
            print(f"Could not load autosave values from file {sav_path}")
            return
        with open(sav_path, "r") as f:
            state = json.load(f)
        for name, value in state.items():
            pv = self._pvs.get(name, None)
            if not pv:
                print(f"{name} is not a valid autosaved PV")
                continue
            pv.set(value)

    def stop(self):
        self._stop_event.set()

    def loop(self):
        if not self._pvs:
            return  # end thread if no PVs to save
        while True:
            self._stop_event.wait(timeout=self.save_period)
            if self._stop_event.is_set(): # Stop requested
                return
            else:  # No stop requested, we should save and continue
                self.save()
