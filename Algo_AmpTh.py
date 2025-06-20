import numpy as np
from Simulations import PhaseTrackerResult, PhaseTrackerStatus
from typing import Tuple
from collections import deque


class PhaseTracker():
    name = 'AmpTh'

    def __init__(
        self,
        threshold_uv: float = -80,
        adaptive: bool = True,
        adaptive_window_sp: int = 1000,
        backoff_sp: int = 1250,
        stim_delay_sp: int = 0,
        interstim_sp: int = 512,
    ):
        """
        Initialize the PhaseTracker with given parameters.

        Args:
            threshold_uv (float): The threshold in microvolts for phase tracking.
            adaptive (bool): Whether to use adaptive thresholding.
        """
        self.threshold_uv = threshold_uv
        self.adaptive = adaptive
        self.adaptive_window_sp = adaptive_window_sp
        self.backoff_sp = backoff_sp
        self.stim_delay_sp = stim_delay_sp
        self.interstim_sp = interstim_sp
        self.target_phase = 0  # static

        self._current_time_sp = 0
        self._last_stim_sp = -np.inf
        self._adaptive_data = deque(maxlen=adaptive_window_sp)

    def update(self, signal: float) -> Tuple[PhaseTrackerResult, dict]:
        """
        Update the PhaseTracker with a new signal value.

        Args:
            signal (float): The new signal value.
        """
        if self.adaptive:
            self._adaptive_data.append(signal)
            current_threshold = np.min(
                (np.min(self._adaptive_data), self.threshold_uv))

        else:
            current_threshold = self.threshold_uv

        self._current_time_sp += 1

        internals = {'current_threshold': current_threshold, 'phase': np.nan}

        if self._current_time_sp - self._last_stim_sp == self.interstim_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.STIM2), internals

        if self._current_time_sp - self._last_stim_sp < self.interstim_sp:
            return PhaseTrackerResult(
                PhaseTrackerStatus.BACKOFF_ISI), internals

        if self._current_time_sp - self._last_stim_sp < self.backoff_sp:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF), internals

        if signal <= current_threshold:
            self._last_stim_sp = self._current_time_sp
            return PhaseTrackerResult(PhaseTrackerStatus.STIM1,
                                      self.stim_delay_sp), internals

        return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE), internals
