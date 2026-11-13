"""The Windows output volume through Core Audio (IAudioEndpointVolume on the default render endpoint), with ctypes.

No package is needed: the COM calls go through ctypes and ole32, which every Windows Python has. soundcard (already
a dependency) opens audio streams through Core Audio but has no endpoint volume, and pywin32 is not a dependency.
Every function runs in the calling thread with its own COM apartment, so callers use ``asyncio.to_thread``. A
failure raises OSError (a failing HRESULT), and the caller falls back to the volume keys.

The interface layouts are Windows' own (mmdeviceapi.h, endpointvolume.h): the vtable slots below are the order of
the methods there, after IUnknown's three.
"""

from __future__ import annotations

import ctypes
import uuid
from ctypes import POINTER, byref, c_float, c_int, c_uint, c_void_p

CLSID_MM_DEVICE_ENUMERATOR = "BCDE0395-E52F-467C-8E3D-C4579291692E"
IID_IMM_DEVICE_ENUMERATOR = "A95664D2-9614-4F35-A746-DE8DB63617E6"
IID_IAUDIO_ENDPOINT_VOLUME = "5CDF2C82-841E-4546-9722-0CF74078229A"
CLSCTX_ALL = 23
E_RENDER, E_MULTIMEDIA = 0, 1
COINIT_MULTITHREADED = 0
RPC_E_CHANGED_MODE = -2147417850  # the thread already has another apartment: use it as it is

# vtable slots
_RELEASE = 2
_GET_DEFAULT_AUDIO_ENDPOINT = 4  # IMMDeviceEnumerator
_ACTIVATE = 3  # IMMDevice
_SET_MASTER_VOLUME_LEVEL_SCALAR = 7  # IAudioEndpointVolume
_GET_MASTER_VOLUME_LEVEL_SCALAR = 9
_SET_MUTE = 14
_GET_MUTE = 15


class _GUID(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    @classmethod
    def of(cls, text: str) -> _GUID:
        guid = cls()
        ctypes.memmove(guid.bytes, uuid.UUID(text).bytes_le, 16)
        return guid


def _method(pointer: c_void_p, slot: int, *argtypes: object):
    """The COM method in ``slot`` of the interface at ``pointer``, as a callable taking the remaining arguments."""
    vtable = ctypes.cast(pointer, POINTER(POINTER(c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)  # raises OSError on a failing HRESULT
    function = prototype(vtable[slot])
    return lambda *args: function(pointer, *args)


def _release(pointer: c_void_p) -> None:
    if pointer:
        vtable = ctypes.cast(pointer, POINTER(POINTER(c_void_p))).contents
        ctypes.WINFUNCTYPE(c_uint, c_void_p)(vtable[_RELEASE])(pointer)


class _Endpoint:
    """The default output device's IAudioEndpointVolume, for one ``with`` block (COM set up and released)."""

    def __enter__(self) -> _Endpoint:
        ole32 = ctypes.OleDLL("ole32")
        try:
            ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
            self._uninit = True
        except OSError as error:
            if getattr(error, "winerror", None) != RPC_E_CHANGED_MODE:
                raise
            self._uninit = False
        self._held: list[c_void_p] = []
        enumerator = c_void_p()
        ole32.CoCreateInstance(
            byref(_GUID.of(CLSID_MM_DEVICE_ENUMERATOR)),
            None,
            CLSCTX_ALL,
            byref(_GUID.of(IID_IMM_DEVICE_ENUMERATOR)),
            byref(enumerator),
        )
        self._held.append(enumerator)
        device = c_void_p()
        _method(enumerator, _GET_DEFAULT_AUDIO_ENDPOINT, c_int, c_int, POINTER(c_void_p))(
            E_RENDER, E_MULTIMEDIA, byref(device)
        )
        self._held.append(device)
        volume = c_void_p()
        _method(device, _ACTIVATE, POINTER(_GUID), c_uint, c_void_p, POINTER(c_void_p))(
            byref(_GUID.of(IID_IAUDIO_ENDPOINT_VOLUME)), CLSCTX_ALL, None, byref(volume)
        )
        self._held.append(volume)
        self.volume = volume
        return self

    def __exit__(self, *_exc: object) -> None:
        for pointer in reversed(self._held):
            _release(pointer)
        if self._uninit:
            ctypes.OleDLL("ole32").CoUninitialize()

    def level(self) -> int:
        scalar = c_float()
        _method(self.volume, _GET_MASTER_VOLUME_LEVEL_SCALAR, POINTER(c_float))(byref(scalar))
        return round(scalar.value * 100)

    def set_level(self, level: int) -> None:
        _method(self.volume, _SET_MASTER_VOLUME_LEVEL_SCALAR, c_float, c_void_p)(max(0, min(100, level)) / 100, None)

    def muted(self) -> bool:
        state = c_int()
        _method(self.volume, _GET_MUTE, POINTER(c_int))(byref(state))
        return bool(state.value)

    def set_muted(self, mute: bool) -> None:
        _method(self.volume, _SET_MUTE, c_int, c_void_p)(1 if mute else 0, None)


def read() -> tuple[int, bool]:
    """(level 0-100, muted) of the default output device."""
    with _Endpoint() as endpoint:
        return endpoint.level(), endpoint.muted()


def set_level(level: int) -> tuple[int, bool]:
    """Set the level, then read it back."""
    with _Endpoint() as endpoint:
        endpoint.set_level(level)
        return endpoint.level(), endpoint.muted()


def step(delta: int) -> tuple[int, bool]:
    """Move the level by ``delta`` points (from what it is now), then read it back."""
    with _Endpoint() as endpoint:
        endpoint.set_level(endpoint.level() + delta)
        return endpoint.level(), endpoint.muted()


def set_muted(mute: bool) -> tuple[int, bool, bool]:
    """Mute or unmute; (level, muted, changed) read back. Asking for the state it is already in changes nothing."""
    with _Endpoint() as endpoint:
        before = endpoint.muted()
        if before != mute:
            endpoint.set_muted(mute)
        return endpoint.level(), endpoint.muted(), before != mute
