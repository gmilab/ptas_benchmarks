import numpy as np
from Simulations import PhaseTrackerResult, PhaseTrackerStatus
from typing import Tuple
from math import nan, pi, isnan
from collections import deque
import scipy.signal


class PhaseTracker():
    name = 'TWave'

    def __init__(self, fs: int, target_phase: float = 0, **kwargs):
        self.ampbuffer = np.zeros(10)
        self.fs = fs

        # internal parameters
        self.last_stim_s = 0.0  # float: units of time elapsed
        self.time_elapsed_s = 0.0  # float: in seconds, relative to end of last block/start of current block

        # store CLAS parameters
        self.amp_threshold_uv = 80
        self.amp_limit_uv = 600
        self.prediction_limit_s = 0.15
        self.backoff_time_s = 5.0
        self.quadrature_thresh = 0.2
        self.quadrature_len_s = 1.0
        self.freq_limits_hz = [0.6, 2.0]

        self.stim2_start_delay_s = 0.6
        self.stim2_end_delay_s = 5.0
        self.stim2_prediction_limit_s = 0.15

        self.high_low_analysis = True
        self.high_freq_vals_hz = [10, 20, 30]
        self.high_low_freq_ratio = 0.15
        self.high_low_freq_lookback_ratio = 0.15
        self.high_low_lookback_nblocks = 5

        # phase target parameters
        target_phase = target_phase % (2 * pi)
        self.target_phase_rad = target_phase
        self.target_phase = target_phase

        # second stim parameters in msec
        self.second_stim_start = nan
        self.second_stim_end = nan

        # buffer
        self.analysis_len_s = 2.0

        # filters
        if fs > 120:
            self.notch_filter = scipy.signal.iirnotch(60, 20, fs=fs)
            self.notch_filter_zi = scipy.signal.lfilter_zi(*self.notch_filter)
        else:
            self.notch_filter = None
            self.notch_filter_zi = None

        self.demean_buffer = deque([0], int(fs * 50))

        # overrides
        for k in kwargs:
            if kwargs[k] is not None:
                print('PhaseTracker: Overriding {:s}:{:s} with {:s}.'.format(
                    k, self.__dict__[k], kwargs[k]))
                self.__dict__[k] = kwargs[k]

        ############################################################
        # initialize phase tracker
        self.data_len_sp = int(fs * 2 * self.analysis_len_s)
        self.data = deque([0.0] * self.data_len_sp, self.data_len_sp)
        self.analysis_sp = int(fs * self.analysis_len_s)
        self.quadrature_sp = int(
            fs * self.quadrature_len_s)  # sp is short for sample

        # construct the wavelet
        M = int(self.analysis_sp)
        w = 5
        s = lambda f: w * fs / (2 * pi * f)

        # set of frequencies, to identify primary freq
        self.wavelet_freqs = np.linspace(self.freq_limits_hz[0],
                                         self.freq_limits_hz[1], 20)

        # create wavelet for each frequency, truncated at the middle
        self.wavelet = [
            PhaseTracker.gen_tmorlet2(M, s(f), w) for f in self.wavelet_freqs
        ]

        # if requested, initialize the high-low frequency ratio
        if self.high_low_analysis:
            self.high_low_data = np.zeros(self.high_low_lookback_nblocks)
            self.high_low_wavelets = [
                PhaseTracker.gen_tmorlet2(M, s(f), w)
                for f in self.high_freq_vals_hz
            ]

    @staticmethod
    def gen_tmorlet2(M: int, s: float, w: float = 5):
        '''
        Generate a truncated complex Morlet wavelet with the given parameters.

        Parameters
        ----------
        M : int
            Length of the wavelet.
        s : float
            Width parameter of the wavelet.
        w : float, optional
            Omega0. Default is 5

        Returns
        -------
        morlet : (M,) ndarray
        '''

        M2 = M * 2
        # M2 = M

        x = np.arange(0, M2) - (M2 - 1.0) / 2
        x = x / s
        wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
        output = np.sqrt(1 / s) * wavelet

        return output[:M]

    def update(self, signal: float):
        ''' 
        Process a block of data through phase tracker and return stimulation status.

        Parameters
        ----------
        signal

        Returns
        -------
        PhaseTrackerResult
            PhaseTrackerResult object containing the stimulation status
        internals
        '''

        blocksize_sp = 1
        blocksize_s = blocksize_sp / self.fs  # length in seconds

        ### adjust time_elapsed accumulator ###
        block_start_time = self.time_elapsed_s
        block_end_time = block_start_time + blocksize_s
        self.time_elapsed_s = block_end_time

        ### apply filters ###
        self.demean_buffer.append(signal)
        signal = signal - np.mean(self.demean_buffer)

        if self.notch_filter is not None:
            signal, self.notch_filter_zi = scipy.signal.lfilter(
                *self.notch_filter, [signal], zi=self.notch_filter_zi)

        if isinstance(signal, np.ndarray):
            signal = signal[0]

        ### roll data buffer ###
        self.data.append(signal)

        if self.second_stim_end < self.time_elapsed_s:
            # go back to normal functioning
            self.second_stim_start = nan
            self.second_stim_end = nan

        ### estimate phase ###
        phase, cfreq, camp, quadrature, hl_ratio = self.estimate()

        # roll amplitude buffer
        self.ampbuffer[:-1] = self.ampbuffer[1:]
        self.ampbuffer[-1] = camp
        meanamp = self.ampbuffer.mean()

        self.high_low_data[:-1] = self.high_low_data[1:]
        self.high_low_data[-1] = hl_ratio
        mean_hl_ratio = self.high_low_data.mean()

        internals = {
            'phase': phase,
            'freq': cfreq,
            'amp': camp,
            'meanamp': meanamp,
            'quadrature': quadrature,
            'hl_ratio': hl_ratio
        }

        status = PhaseTrackerStatus.NONE

        # check if we're waiting for the 2nd stim
        # if NOT, run normal checks
        if isnan(self.second_stim_start):
            ### check backoff criteria ###
            if ((self.last_stim_s + self.backoff_time_s)
                    > (self.time_elapsed_s + self.prediction_limit_s)):
                status = PhaseTrackerStatus(
                    status) | PhaseTrackerStatus.BACKOFF

            ### check amplitude criteria ###
            if (meanamp < self.amp_threshold_uv) or (meanamp
                                                     > self.amp_limit_uv):
                status = PhaseTrackerStatus(
                    status) | PhaseTrackerStatus.INHIBITED_AMP

            ### check quadrature ###
            if (quadrature is not None) and (quadrature
                                             < self.quadrature_thresh):
                status = PhaseTrackerStatus(
                    status) | PhaseTrackerStatus.INHIBITED_QUADRATURE

            if self.high_low_analysis and (
                (mean_hl_ratio > self.high_low_freq_lookback_ratio) or
                (hl_ratio > self.high_low_freq_ratio)):
                status = PhaseTrackerStatus(
                    status) | PhaseTrackerStatus.INHIBITED_RATIO

            if status != PhaseTrackerStatus.NONE:
                # if we're inhibited, don't do anything
                return PhaseTrackerResult(status), internals

        # if we are waiting for 2nd stim, but before the backoff window, only use phase targeting
        if self.time_elapsed_s < self.second_stim_start:
            return PhaseTrackerResult(
                PhaseTrackerStatus.BACKOFF_ISI), internals

        ### perform forward prediction ###
        delta_t = ((self.target_phase_rad - phase - pi / 12) %
                   (2 * pi)) / (cfreq * 2 * pi)

        # cue a stim for the next target phase
        if isnan(self.second_stim_start):
            if delta_t > self.prediction_limit_s:
                return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE,
                                          int(delta_t * self.fs)), internals

            self.last_stim_s = self.time_elapsed_s + delta_t  # update stim time to compute backoff
            self.second_stim_start = self.last_stim_s + self.stim2_start_delay_s  # update
            self.second_stim_end = self.last_stim_s + self.stim2_end_delay_s

            return PhaseTrackerResult(PhaseTrackerStatus.STIM1,
                                      int(delta_t * self.fs)), internals

        else:
            if delta_t > self.stim2_prediction_limit_s:
                return PhaseTrackerResult(PhaseTrackerStatus.WRONGPHASE,
                                          int(delta_t * self.fs)), internals

            self.second_stim_start = nan
            self.second_stim_end = nan

            return PhaseTrackerResult(PhaseTrackerStatus.STIM2,
                                      int(delta_t * self.fs)), internals

    def estimate(self) -> Tuple[float, float, float, float, float]:
        '''
        Return estimated phase / amplitude / frequency / quadrature at most recent point of the internal buffer using the wavelet transform.

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
        hl_ratio = np.nan

        # the data that we're analyzing
        cdata = np.array(self.data)[-self.analysis_sp:]

        # convolve the list of wavelets
        conv_vals = [np.dot(cdata, w) for w in self.wavelet]

        # choose the one with highest amp/phase
        amp_conv_vals = np.abs(conv_vals)
        amp_max = np.argmax(amp_conv_vals)

        # create outputs
        amp = amp_conv_vals[amp_max] / 2
        freq = self.wavelet_freqs[amp_max]
        # phase = np.angle(conv_vals[amp_max])
        phase = np.arctan2(np.real(conv_vals[amp_max]),
                           np.imag(conv_vals[amp_max])) - (pi / 2)
        phase = phase % (2 * pi)

        ### high low ratio ###
        # convolve the list of wavelets
        conv_vals_hl = [np.dot(cdata, w) for w in self.high_low_wavelets]

        # get average amplitude
        hf_amp = np.mean(np.abs(conv_vals_hl))

        # compute ratio and store
        hl_ratio = hf_amp / np.mean(np.abs(conv_vals))

        ### determine if we're locked on ###
        est_phase = (np.arange(self.quadrature_sp) / self.fs) * freq * 2 * pi
        est_phase = est_phase - est_phase[-1] + phase
        est_sig = np.cos(est_phase)
        est_sig = est_sig / np.trapz(np.abs(est_sig)) * est_sig.size

        # normalize the signal
        normsig = cdata[-self.quadrature_sp:] / np.trapz(
            np.abs(cdata[-self.quadrature_sp:])) * cdata.size
        quadrature = np.trapz(normsig * est_sig) / cdata.size

        return phase, freq, amp, quadrature, hl_ratio
