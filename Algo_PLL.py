import numpy as np
import scipy.signal
from Simulations import PhaseTrackerResult, PhaseTrackerStatus
from Inhibitors import MinAmp
from typing import Tuple, List, Optional
from collections import deque


class PhaseTracker():
    name = 'PLL'

    def __init__(
        self,
        fs: int,
        KPLL: float = 10,
        backoff_s: float = 5.0,
        target_phase_tolerance_rad: Optional[
            float] = None,  # default is configured below
        target_phase: float = 0,
        rms_window_length_s: float = 2,
        inhibitors: Optional[List] = None,
    ):
        self.fs = fs
        self.current_time_sp = 0
        self.pll_phi = 0
        self.KPLL = KPLL

        self.last_stim_sp = 0
        self.backoff_sp = int(backoff_s * fs)
        self.stim_count = 0

        self.backoff_interstim_sp = int(0.5 * fs)

        self.inhibitors = inhibitors or [MinAmp(int(fs * 2.0))]

        self.target_phase = target_phase
        self.target_phase_tolerance_rad = target_phase_tolerance_rad or (
            4 * np.pi / fs)

        # self.loop_gain_filter = scipy.signal.butter(2,
        #                                             1,
        #                                             'lowpass',
        #                                             fs=fs,
        #                                             output='ba')
        # self.loop_gain_filter_zi = scipy.signal.lfilter_zi(
        #     *self.loop_gain_filter)

        rms_window_length_sp = int(rms_window_length_s * fs)
        self.window = deque([1], maxlen=rms_window_length_sp)

    def update(self, signal: float) -> Tuple[PhaseTrackerResult, dict]:
        """
        Estimate the phase of the input signal using a first-order PLL, updating state with only the latest sample.

        Args:
            signal (float): The input signal sample to be processed.

        Returns:
            PhaseTrackerResult
        """
        fs = self.fs
        KPLL = self.KPLL  # PLL gain (tune this parameter)

        self.window.append(signal)
        mean_rms = np.sqrt(np.mean(np.square(self.window)))

        # Compute base phase using persistent sample index
        base_phase = 2 * np.pi * (self.current_time_sp / fs)

        # Compute NCO output with current phase correction
        nco_output = np.cos(base_phase + self.pll_phi)

        # Compute phase error (multiplication of input and NCO output)
        error = signal * nco_output / mean_rms / fs

        # error, self.loop_gain_filter_zi = scipy.signal.lfilter(
        #     *self.loop_gain_filter, [error], zi=self.loop_gain_filter_zi)
        # error = error[0]

        # Update persistent phase offset
        self.pll_phi = self.pll_phi - KPLL * error

        # Compute instantaneous phase estimate (mod 2π)
        estimated_phase = (base_phase + self.pll_phi + np.pi / 2) % (2 * np.pi)
        # Increment sample counter (ensures continuous phase tracking)
        self.current_time_sp += 1

        internals = {
            'phase': estimated_phase,
            'error': error,
            'pll_phi': self.pll_phi,
            'pll_base_phase': base_phase,
            'mean_rms': mean_rms,
        }

        status = PhaseTrackerStatus.NONE

        if self.stim_count == 0 and (self.current_time_sp
                                     < (self.backoff_sp + self.last_stim_sp)):
            status = PhaseTrackerStatus(status) | PhaseTrackerStatus.BACKOFF

        if self.stim_count == 1 and (self.current_time_sp < (
                self.backoff_interstim_sp + self.last_stim_sp)):
            status = PhaseTrackerStatus(
                status) | PhaseTrackerStatus.BACKOFF_ISI

        for inh in self.inhibitors:
            # Update the inhibitor with the current signal
            inh_status = inh.update(signal)
            if not inh_status:
                status = PhaseTrackerStatus(
                    status) | PhaseTrackerStatus.INHIBITED
            internals[inh.__name__] = inh_status

        if status != PhaseTrackerStatus.NONE:
            return PhaseTrackerResult(status), internals

        if np.abs((estimated_phase - self.target_phase) %
                  (2 * np.pi)) < self.target_phase_tolerance_rad:
            self.stim_count += 1
            self.last_stim_sp = self.current_time_sp

            if self.stim_count == 1:
                return PhaseTrackerResult(PhaseTrackerStatus.STIM1), internals

            elif self.stim_count == 2:
                self.stim_count = 0
                return PhaseTrackerResult(PhaseTrackerStatus.STIM2), internals

        return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE), internals
