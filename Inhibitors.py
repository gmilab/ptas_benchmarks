from collections import deque
import numpy as np
import scipy.signal
from typing import Literal


class MinAmp():
    """
    A class to monitor the minimum amplitude of a signal over a specified window length.
    """
    __name__ = 'MinAmp'

    def __init__(self,
                 window_length_sp: int,
                 min_amp_threshold_uv: float = 75):
        self.window = deque(maxlen=window_length_sp)
        self.min_amp_threshold = min_amp_threshold_uv

    def update(self, signal: float) -> bool:
        """
        Update the buffer, and return whether we meet the minimum amplitude criteria.

        Args:
            signal (float): The input signal sample to be processed.

        Returns:
            bool: True if the maximum amplitude exceeds the threshold, False otherwise.
        """
        self.window.append(signal)
        return np.nanmax(np.abs(self.window)) > self.min_amp_threshold


class MaxAmp():
    """
    A class to monitor the maximum amplitude of a signal over a specified window length.
    """
    __name__ = 'MaxAmp'

    def __init__(self,
                 window_length_sp: int,
                 max_amp_threshold_uv: float = 75):
        self.window = deque(maxlen=window_length_sp)
        self.max_amp_threshold = max_amp_threshold_uv

    def update(self, signal: float) -> bool:
        """
        Update the buffer, and return whether we meet the maximum amplitude criteria.

        Args:
            signal (float): The input signal sample to be processed.

        Returns:
            bool: True if the maximum amplitude remains below the threshold, False otherwise.
        """
        self.window.append(signal)
        return np.nanmax(np.abs(self.window)) < self.max_amp_threshold


class HLRatio():
    __name__ = 'HLRatio'

    def __init__(
        self,
        fs: float,
        hl_ratio_threshold: float = -2.0,
        hl_ratio_window_s: float = 0.5,
        lowband: tuple = (1, 4),
        highband: tuple = (12, 80),
        method: Literal['hilbert', 'rms'] = 'rms',
    ):
        self.fs = fs
        self.hl_ratio_threshold = hl_ratio_threshold

        # Initialize the HL ratio buffer
        self.buffer_size = int(fs * hl_ratio_window_s)
        self.method = method

        self.sos_low = scipy.signal.butter(4,
                                           lowband,
                                           btype='bandpass',
                                           output='sos',
                                           fs=fs)
        self.sos_high = scipy.signal.butter(4,
                                            highband,
                                            btype='bandpass',
                                            output='sos',
                                            fs=fs)

        self.zi_low = scipy.signal.sosfilt_zi(self.sos_low)
        self.zi_high = scipy.signal.sosfilt_zi(self.sos_high)

        self.high_buffer = deque(maxlen=self.buffer_size)
        self.low_buffer = deque(maxlen=self.buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)

    def update(self, signal: float) -> bool:
        """
        Update the HL ratio with a new signal sample.

        Args:
            signal (float): The input signal sample to be processed.

        Returns:
            bool: True if HL ratio exceeds the threshold, False otherwise.
        """
        # Apply bandpass filter
        filtered_low, self.zi_low = scipy.signal.sosfilt(
            self.sos_low, signal, self.zi_low)
        filtered_high, self.zi_high = scipy.signal.sosfilt(
            self.sos_high, signal, self.zi_high)

        self.high_buffer.append(filtered_high)
        self.low_buffer.append(filtered_low)

        if self.method == 'rms':
            # Compute RMS
            power_lf = np.sqrt(np.mean(np.square(self.low_buffer)))
            power_hf = np.sqrt(np.mean(np.square(self.high_buffer)))

        elif self.method == 'hilbert':
            # Compute Hilbert envelope
            envelope_lf = np.abs(scipy.signal.hilbert(self.low_buffer))
            envelope_hf = np.abs(scipy.signal.hilbert(self.high_buffer))
            power_lf = envelope_lf**2
            power_hf = envelope_hf**2

        hl_ratio = np.mean(power_hf) / np.mean(power_lf)
        hl_ratio = np.log10(hl_ratio)

        # Update buffer and check threshold
        self.buffer.append(hl_ratio)
        return hl_ratio < self.hl_ratio_threshold
