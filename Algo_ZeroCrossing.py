import numpy as np
from Simulations import PhaseTrackerResult, PhaseTrackerStatus
from typing import Tuple, List, Optional
from math import nan, pi, isnan
from collections import deque
import scipy.signal


class PhaseTracker():
    name = 'ZeroCrossing'

    def __init__(
        self,
        fs: int = 256,  # sampling frequency in Hz (updated)
        min_peak_uv: float = -40,
        min_interval_ms: int = 200,
        max_interval_ms: int = 800,
        backoff_sp: int = 1250,
        stim_delay_sp: int = 0,  # delay after positive zero-crossing
        interstim_sp: int = 1000,  # 1s between tones
        history_len: int = 2000,   # buffer for 2s
    ):
        """
        Zero-crossing based slow-wave detector and stimulator.
        """
        self.fs = fs
        self.min_peak_uv = min_peak_uv
        self.min_interval_sp = int(min_interval_ms * fs / 1000)
        self.max_interval_sp = int(max_interval_ms * fs / 1000)
        self.backoff_sp = backoff_sp
        self.stim_delay_sp = stim_delay_sp
        self.interstim_sp = interstim_sp
        self._current_time_sp = 0
        self._last_stim_sp = -np.inf
        self._signal_history = deque(maxlen=history_len)
        self._last_value = None
        self._negzc_index = None
        self._negzc_time = None
        self._neg_peak = None
        self._neg_peak_time = None
        self._awaiting_poszc = False

        self.target_phase = 0  # static

    def update(self, signal: float) -> Tuple[PhaseTrackerResult, dict]:
        self._signal_history.append(signal)
        self._current_time_sp += 1
        internals = {'phase': np.nan}

        # ISI and backoff logic
        if self._current_time_sp - self._last_stim_sp == self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.STIM2), internals
        if self._current_time_sp - self._last_stim_sp < self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF_ISI), internals
        if self._current_time_sp - self._last_stim_sp < self.backoff_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF), internals

        # Zero-crossing detection
        if self._last_value is not None:
            # Negative-going zero-crossing
            if self._last_value > 0 and signal <= 0:
                self._negzc_index = len(self._signal_history) - 2
                self._negzc_time = self._current_time_sp - 2
                self._neg_peak = signal
                self._neg_peak_time = self._current_time_sp - 1
                self._awaiting_poszc = True
            elif self._awaiting_poszc:
                # Track most negative peak after neg-zc
                if signal < self._neg_peak:
                    self._neg_peak = signal
                    self._neg_peak_time = self._current_time_sp - 1
                # Positive-going zero-crossing
                if self._last_value < 0 and signal >= 0:
                    poszc_time = self._current_time_sp - 1
                    interval = poszc_time - self._negzc_time
                    # Check slow-wave criteria
                    if (
                        self._neg_peak <= self.min_peak_uv and
                        self.min_interval_sp <= interval <= self.max_interval_sp
                    ):
                        # Valid slow-wave detected
                        self._last_stim_sp = self._current_time_sp
                        self._awaiting_poszc = False
                        internals.update({
                            'negzc_time': self._negzc_time,
                            'neg_peak': self._neg_peak,
                            'neg_peak_time': self._neg_peak_time,
                            'poszc_time': poszc_time,
                            'interval': interval
                        })
                        return PhaseTrackerResult(PhaseTrackerStatus.STIM1, self.stim_delay_sp), internals
                    else:
                        # Reset and wait for next neg-zc
                        self._awaiting_poszc = False
        self._last_value = signal
        return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE), internals
