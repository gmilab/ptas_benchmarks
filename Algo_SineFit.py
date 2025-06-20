import numpy as np
import scipy.signal
from collections import deque
from Simulations import PhaseTrackerResult, PhaseTrackerStatus
from typing import Tuple
from math import pi, nan, isnan
from scipy.optimize import curve_fit


class PhaseTracker():
    name = 'SineFit'

    def __init__(self, fs: int, target_phase: float = 0, **kwargs):
        self.fs = fs
        self.target_phase = target_phase % (2 * pi)
        self.analysis_len_s = 10.0
        self.buffer = deque([0.0] * int(fs * self.analysis_len_s),
                            maxlen=int(fs * self.analysis_len_s))
        self.soband = (0.5, 2.0)
        self.bandwidth = 0.5
        self.power_threshold = 0.6
        self.fit_error_threshold = 0.1
        self.max_prediction_s = 0.1
        self.backoff_time_s = 5.0

        self.last_stim_s = 0.0
        self.time_elapsed_s = 0.0

        for k in kwargs:
            if kwargs[k] is not None:
                print(f"Overriding {k}: {self.__dict__[k]} -> {kwargs[k]}")
                self.__dict__[k] = kwargs[k]

        self.data_len_sp = int(fs * 2 * self.analysis_len_s)
        self.data = deque([0.0] * self.data_len_sp, self.data_len_sp)
        self.analysis_sp = int(fs * self.analysis_len_s)

    def update(self, signal: float):
        blocksize_sp = 1
        blocksize_s = blocksize_sp / self.fs

        ### adjust time_elapsed accumulator ###
        block_start_time = self.time_elapsed_s
        block_end_time = block_start_time + blocksize_s
        self.time_elapsed_s = block_end_time

        self.data.append(signal)
        t = np.arange(
            self.analysis_sp
        ) / self.fs - 10  # time vector relative to the start of the data

        if self.time_elapsed_s < self.analysis_len_s:
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF), {
                'phase': nan,
                'fit_error': nan
            }

        cfreq, fit_error, hl_ratio, popt = self.estimate()

        if hl_ratio < self.power_threshold:
            return PhaseTrackerResult(PhaseTrackerStatus.INHIBITED_RATIO), {
                'phase': nan,
                'fit_error': nan
            }

        if fit_error > self.fit_error_threshold:
            return PhaseTrackerResult(
                PhaseTrackerStatus.INHIBITED_QUADRATURE), {
                    'phase': nan,
                    'fit_error': fit_error
                }

        pred_phase = (2 * np.pi * cfreq * t[-1] + popt[1]) % (2 * pi)
        phase_diff = (self.target_phase - pred_phase) % (2 * pi)
        time_to_target = phase_diff / (2 * np.pi * cfreq)

        if (self.last_stim_s + self.backoff_time_s) > (self.time_elapsed_s +
                                                       time_to_target):
            return PhaseTrackerResult(PhaseTrackerStatus.BACKOFF), {
                'phase': pred_phase,
                'fit_error': fit_error
            }

        if (time_to_target > self.max_prediction_s):
            return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE,
                                      int(time_to_target * self.fs)), {
                                          'phase': pred_phase,
                                          'fit_error': fit_error
                                      }

        self.last_stim_s = self.time_elapsed_s + time_to_target
        return PhaseTrackerResult(PhaseTrackerStatus.STIM1,
                                  int((time_to_target) * self.fs)), {
                                      'phase': pred_phase,
                                      'fit_error': fit_error
                                  }

    def estimate(self) -> Tuple[float, float, float, float, float]:
        '''
        Return estimated phase / amplitude / frequency / fit error.

        If block is provided, this function also rolls the internal buffer and appends the data from block.

        Parameters
        ----------
        block : np.ndarray (Default: None)
            New data

        Returns
        -------
        phase : float
            The current estimated phase

        freq : float
            The current estimated frequency

        amp : float
            The current estimated amplitude

        quadrature : float

        '''
        # the data that we're analyzing
        cdata = np.array(self.data)[-self.analysis_sp:]
        t = np.arange(len(cdata)) / self.fs - 10

        # FFT to detect center frequency
        freqs = np.fft.rfftfreq(len(cdata), 1 / self.fs)
        fft_vals = np.fft.rfft(cdata)
        so_mask = (freqs >= self.soband[0]) & (freqs <= self.soband[1])
        so_power = np.sum(np.abs(fft_vals[so_mask])**2)
        total_power = np.sum(np.abs(fft_vals)**2)
        hl_ratio = so_power / total_power

        cfreq = freqs[so_mask][np.argmax(np.abs(fft_vals[so_mask]))]
        low = max(cfreq - self.bandwidth, 0.01)
        high = cfreq + self.bandwidth
        b, a = scipy.signal.butter(1, [low, high], btype='band', fs=self.fs)
        filtered = scipy.signal.filtfilt(b, a, cdata)

        # Hilbert transform
        analytic = scipy.signal.hilbert(filtered)
        phase_signal = np.real(analytic)

        # Fit window
        fit_start = int(0.8 * len(cdata))
        fit_end = int(0.95 * len(cdata))

        fit_t = t[fit_start:fit_end]
        fit_y = phase_signal[fit_start:fit_end]

        def sine_func(x, A, phi):
            return A * np.cos(2 * np.pi * cfreq * x + phi)

        popt, _ = curve_fit(sine_func, fit_t, fit_y, p0=[np.std(fit_y), 0])
        if popt[0] < 0:
            popt[0] = -popt[0]
            popt[1] = (popt[1] + np.pi) % (2 * np.pi)

        fit_error = np.mean(
            (sine_func(fit_t, *popt) - fit_y)**2) / np.mean(fit_y**2)

        return cfreq, fit_error, hl_ratio, popt
