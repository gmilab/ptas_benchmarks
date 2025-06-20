import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
from math import pi
import enum
from typing import NamedTuple, Optional, List, Tuple, Literal
from tqdm import tqdm

class PhaseTrackerStatus(enum.IntFlag):
    """
    Enum class to represent the result of phase tracking.
    """
    NONE = 0

    STIM1 = 1
    STIM2 = 2

    INHIBITED = 4
    INHIBITED_AMP = 8
    INHIBITED_RATIO = 16
    INHIBITED_QUADRATURE = 32

    WRONGPHASE = 64
    BACKOFF = 128
    BACKOFF_ISI = 256


class PhaseTrackerResult(NamedTuple):
    status: PhaseTrackerStatus
    delay_sp: int = 0


class SimulationDataset(NamedTuple):
    t: np.ndarray
    signal: np.ndarray
    fs: float
    name: str

    def compute_true_phase(
        self, filt_bandpass_hz: Optional[tuple] = (0.5, 4.0)) -> np.ndarray:
        """
        Compute the true phase of the signal using Hilbert transform.
        """
        if filt_bandpass_hz is not None:
            b, a = scipy.signal.butter(2,
                                       filt_bandpass_hz,
                                       'bandpass',
                                       fs=self.fs)
            filtered_signal = scipy.signal.filtfilt(b, a, self.signal)
        else:
            filtered_signal = self.signal

        analytic_signal = scipy.signal.hilbert(filtered_signal)
        signal_phase = np.angle(analytic_signal) % (2 * pi)

        return signal_phase


class SimulationResult():

    def __init__(self, Dataset: SimulationDataset, PhaseTracker: object,
                 stims_sp: list, status_ts: List[PhaseTrackerStatus],
                 internals_ts: List[dict]):
        self.Dataset = Dataset
        self.PhaseTracker = PhaseTracker
        self.stims_sp = stims_sp
        self.status_ts = status_ts
        self.internals_ts = internals_ts

        self.computed_stim_phases = {}

    def plot_timeseries(
        self,
        axis_kwargs: Optional[dict] = None,
        time_lim: Optional[tuple] = None,
    ):
        """
        Plot the time series of the signal and the phase tracker status, applying
        the time_mask to all plotting calls.
        """
        # Define a time_mask that covers the whole data if time_lim is not provided.
        if time_lim is not None:
            time_mask = (self.Dataset.t >= time_lim[0]) & (self.Dataset.t
                                                           <= time_lim[1])
        else:
            time_mask = slice(None)  # This allows slicing the full array

        # Design the filter.
        butter_filt = scipy.signal.butter(2, [0.5, 4],
                                          'bandpass',
                                          fs=self.Dataset.fs)

        # Filter only the data within the time range.
        filtered_signal = scipy.signal.filtfilt(butter_filt[0], butter_filt[1],
                                                self.Dataset.signal[time_mask])

        # Create the figure and axes.
        fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

        # Plot the raw signal and the filtered signal.
        tax = ax[0]
        tax.plot(self.Dataset.t[time_mask],
                 self.Dataset.signal[time_mask],
                 color='black',
                 lw=0.1,
                 alpha=0.8)
        tax.plot(self.Dataset.t[time_mask],
                 filtered_signal,
                 color='tab:blue',
                 label='Filtered Signal')
        tax.set_title(f'{self.PhaseTracker.name} - {self.Dataset.name}')
        tax.set_ylabel('Signal')
        tax.grid()
        if axis_kwargs:
            tax.set(**axis_kwargs)

        # For the estimated phase, filter the internals_ts based on the time mask.
        # Assumes that self.internals_ts is aligned with self.Dataset.t.
        internals_ts_masked = [
            entry for idx, entry in enumerate(self.internals_ts)
            if (time_mask if isinstance(time_mask, slice) else time_mask[idx])
        ]
        tax = ax[1]
        tax.plot(self.Dataset.t[time_mask],
                 [entry['phase'] % (2 * pi) for entry in internals_ts_masked])
        tax.set_title('Estimated Phase')
        tax.set_ylabel('Phase (rad)')
        tax.set_ylim(0, 2 * pi)
        tax.set_yticks([0, pi / 2, pi, 3 * pi / 2, 2 * pi])
        tax.set_yticklabels([
            '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'
        ])
        tax.grid()
        if axis_kwargs:
            tax.set(**axis_kwargs)

        # For the phase tracker status, filter status_ts based on the time mask.
        # Assumes that self.status_ts is aligned with self.Dataset.t.
        if isinstance(time_mask, slice):
            status_ts_masked = self.status_ts[time_mask]
        else:
            status_ts_masked = [
                self.status_ts[idx] for idx in range(len(self.Dataset.t))
                if time_mask[idx]
            ]

        tax = ax[2]
        tax.scatter(x=self.Dataset.t[time_mask],
                    y=status_ts_masked,
                    marker='.')
        tax.set_yscale('log')
        yticks, yticklabels = zip(*[(s.value, s.name)
                                    for s in PhaseTrackerStatus])
        tax.set_yticks(yticks)
        tax.set_yticklabels(yticklabels)
        tax.set_title('Phase Tracker Status')
        tax.set_ylabel('Status')
        tax.set_xlabel('Time (s)')
        tax.grid()
        if axis_kwargs:
            tax.set(**axis_kwargs)

        return fig

    def compute_stim_phase(
        self,
        filt_bandpass_hz: Optional[tuple] = None,
    ):
        if self.computed_stim_phases.get(filt_bandpass_hz) is not None:
            return self.computed_stim_phases[filt_bandpass_hz]

        else:
            phase_kwargs = {}
            if filt_bandpass_hz is not None:
                phase_kwargs['filt_bandpass_hz'] = filt_bandpass_hz
            signal_phase = self.Dataset.compute_true_phase(**phase_kwargs)

            # phase of signal at stim_trigs
            stim_phases = signal_phase[self.stims_sp]

            # Store the computed stim phases for future use
            self.computed_stim_phases[filt_bandpass_hz] = stim_phases

            return stim_phases

    # Test accuracy and precision of phase estimation
    def plot_phase_hist(
        self,
        filt_bandpass_hz: Optional[tuple] = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot the phase histogram of the signal at the stimulation triggers.
        """
        # phase of signal at stim_trigs
        stim_phases = self.compute_stim_phase(
            filt_bandpass_hz=filt_bandpass_hz)

        hf = plot_phase_hist_array(
            stim_phases,
            target_phase=self.PhaseTracker.target_phase,
            title=f'{self.PhaseTracker.name} - {self.Dataset.name}')

        return hf

    def compute_timelocked(
        self,
        time_window_s: Optional[Tuple[float, float]] = None,
        filt_bandpass_hz: Optional[tuple] = None,
    ):
        """
        Compute the time-locked signal around the stimulation triggers.
        Used for plotting the evoked response.
        """
        time_window_s = time_window_s or (-3.0, 2.0)
        filt_bandpass_hz = filt_bandpass_hz or (0.5, 4.0)

        # Define the time window around the stim triggers
        time_window_sp = (int(time_window_s[0] * self.Dataset.fs),
                          int(time_window_s[1] * self.Dataset.fs))
        time_axis = np.arange(time_window_sp[0],
                              time_window_sp[1]) / self.Dataset.fs

        b, a = scipy.signal.butter(2,
                                   filt_bandpass_hz,
                                   'bandpass',
                                   fs=self.Dataset.fs)
        filtered = scipy.signal.filtfilt(b, a, self.Dataset.signal)

        # Create an array to hold the evoked responses
        evoked_responses = []
        for stim in self.stims_sp:
            # Get the time window around the stimulation trigger
            start = stim + time_window_sp[0]
            end = stim + time_window_sp[1]

            # Check if the indices are within bounds
            if start >= 0 and end < len(filtered):
                evoked_responses.append(filtered[start:end])
            else:
                continue

        # Convert the list to a 2D array
        evoked_responses = np.array(evoked_responses)

        return evoked_responses, time_axis

    def plot_evoked(
        self,
        time_window_s: Optional[Tuple[float, float]] = None,
        filt_bandpass_hz: Optional[tuple] = None,
        num_bootstrap: int = 1000,
    ):
        """
        Plot the evoked response of the signal around the stimulation triggers.
        """

        # Compute the evoked response
        evoked_responses, time_axis = self.compute_timelocked(
            time_window_s=time_window_s,
            filt_bandpass_hz=filt_bandpass_hz,
        )

        # Compute the average evoked response
        avg_evoked = np.mean(evoked_responses, axis=0)

        # also compute the boostrapped 95th percentile confidence interval
        # for the evoked response using scipy
        statistic = lambda data: np.mean(data, axis=0)
        ci = scipy.stats.bootstrap(
            (evoked_responses, ),
            statistic,
            confidence_level=0.95,
            n_resamples=num_bootstrap,
            method='basic',
            axis=0,
        )

        # Plot the average evoked response
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_axis, evoked_responses.T, color='gray', alpha=0.2)
        ax.fill_between(
            time_axis,
            ci.confidence_interval.low,
            ci.confidence_interval.high,
            color='tab:blue',
            alpha=0.6,
            label='95% CI',
        )
        ax.plot(time_axis, avg_evoked, color='tab:blue', lw=2)
        ax.set_title(f'Evoked Response - {self.Dataset.name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')
        ax.grid()

        # add a note in the top left corner inside the axis box with N
        ax.text(0.05,
                0.95,
                f'N={len(self.stims_sp)}',
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        # set the y-axis to the 90th percentile of the evoked responses max
        max_vals = np.max(np.abs(evoked_responses), axis=1)
        max_val = np.percentile(max_vals, 90)
        max_val = 200
        ax.set_ylim(-max_val, max_val)

        return fig

    def plot_internals(self):
        plot_data = {
            'PLL': [
                {
                    'data': lambda i: i['phase'] % (2 * pi),
                    'ylabel': 'Final Est Phase (rad)',
                    'yticks': [0, pi, 2 * pi],
                    'yticklabels': ['0', r'$\pi$', r'$2\pi$']
                },
                {
                    'data': lambda i: i['pll_base_phase'] % (2 * pi),
                    'ylabel': 'Base Phase',
                    'yticks': [0, pi, 2 * pi],
                    'yticklabels': ['0', r'$\pi$', r'$2\pi$']
                },
                {
                    'data': lambda i: i['pll_phi'] % (2 * pi),
                    'ylabel': 'PLL Offset Accumulator (rad/$\\pi$)',
                    'yticks': [0, pi, 2 * pi],
                    'yticklabels': ['0', r'$\pi$', r'$2\pi$']
                },
                {
                    'data': lambda i: i['mean_rms'] % (2 * pi),
                    'ylabel': 'Estimated amplitude',
                },
            ],
            'AmpTh': [{
                'data': lambda i: i['current_threshold'],
                'ylabel': 'Current Adaptive Threshold (uV)',
            }],
            'ZeroCrossing': [{
                    'data': lambda i: i['negzc_time'],
                    'ylabel': 'Negative ZC Time (samples)'
                },
                {
                    'data': lambda i: i['neg_peak'],
                    'ylabel': 'Negative Peak Amplitude (µV)'
                },
                {
                    'data': lambda i: i['neg_peak_time'],
                    'ylabel': 'Negative Peak Time (samples)'
                },
                {
                    'data': lambda i: i['poszc_time'],
                    'ylabel': 'Positive ZC Time (samples)'
                },
                {
                    'data': lambda i: i['interval'],
                    'ylabel': 'Interval (samples)'
                }],
            'SineFit': [
                {
                    'data': lambda i: i['phase'] % (2 * pi),
                    'ylabel': 'Final Est Phase (rad)',
                    'yticks': [0, pi, 2 * pi],
                    'yticklabels': ['0', r'$\pi$', r'$2\pi$']
                },
                {
                    'data': lambda i: i['fit_error'],
                    'ylabel': 'Fit Error',
                },
                {
                    'data': lambda i: i['fit_phase'] % (2 * pi),
                    'ylabel': 'Fit Phase',
                    'yticks': [0, pi, 2 * pi],
                    'yticklabels': ['0', r'$\pi$', r'$2\pi$']
                }],
            'TWave': [
                {
                    'data': lambda i: i['phase'] % (2 * pi),
                    'ylabel': 'Final Est Phase (rad)',
                    'yticks': [0, pi, 2 * pi],
                    'yticklabels': ['0', r'$\pi$', r'$2\pi$']
                },
                {
                    'data': lambda i: i['quadrature'],
                    'ylabel': 'Quadrature',
                    'hline': lambda rslt: rslt.PhaseTracker.quadrature_thresh,
                },
                {
                    'data': lambda i: i['hl_ratio'],
                    'ylabel': 'High-Low Ratio',
                    'hline':
                    lambda rslt: rslt.PhaseTracker.high_low_freq_ratio,
                },
                {
                    'data': lambda i: i['amp'],
                    'ylabel': 'SWS amplitude (uV)'
                },
            ]
        }

        if self.PhaseTracker.name not in plot_data:
            raise ValueError(
                f'PhaseTracker {self.PhaseTracker.name} not supported')
        tpd = plot_data[self.PhaseTracker.name]

        # Create the figure and axes
        ny = len(tpd) + 1
        fig, ax = plt.subplots(ny, 1, figsize=(16, ny * 3), sharex=True)
        tax = ax[0]
        tax.plot(self.Dataset.t, self.Dataset.signal)
        tax.set_ylabel('Signal')
        tax.grid()

        for haxi, plot in enumerate(tpd):
            tax = ax[haxi + 1]
            tax.plot(self.Dataset.t,
                     [plot['data'](i) for i in self.internals_ts])
            tax.set_ylabel(plot['ylabel'])
            if 'hline' in plot:
                tax.axhline(y=plot['hline'](self), color='red', linestyle='--')
            if 'yticks' in plot:
                tax.set_yticks(plot['yticks'])
                tax.set_yticklabels(plot['yticklabels'])
            tax.grid()

        tax.set_xlabel('Time (s)')

        return fig


class SimulationGroupResult():

    def __init__(self, results: List[SimulationResult], name: str = ''):
        self.results = results
        self.n = len(results)

        # check if all the phase trackers match
        phase_trackers = [result.PhaseTracker.name for result in results]
        if len(set(phase_trackers)) != 1:
            raise ValueError(
                'All results must have the same phase tracker for comparison.')
        self.phase_tracker_name = phase_trackers[0]

        # check if all the target phases match
        target_phases = [
            result.PhaseTracker.target_phase for result in results
        ]
        if len(set(target_phases)) != 1:
            raise ValueError(
                'All results must have the same target phase for comparison.')
        self.target_phase = target_phases[0]

        self.name = name
        self.computed_stim_phases = {}

    def compute_stim_phase(
        self,
        filt_bandpass_hz: Optional[tuple] = None,
    ):
        if self.computed_stim_phases.get(filt_bandpass_hz) is not None:
            return self.computed_stim_phases[filt_bandpass_hz]

        else:
            stim_phases = []
            for result in self.results:
                phase_kwargs = {}
                if filt_bandpass_hz is not None:
                    phase_kwargs['filt_bandpass_hz'] = filt_bandpass_hz
                stim_phase = result.compute_stim_phase(**phase_kwargs)
                stim_phases.extend(stim_phase)

            stim_phases = np.array(stim_phases)
            self.computed_stim_phases[filt_bandpass_hz] = stim_phases

            return stim_phases

    def plot_phase_hist(self, filt_bandpass_hz: Optional[tuple] = None):
        '''
        Plot the phase histograms for all results in the group.
        '''
        stim_phases = self.compute_stim_phase(
            filt_bandpass_hz=filt_bandpass_hz)

        # plot the histogram
        hf = plot_phase_hist_array(
            stim_phases,
            target_phase=self.target_phase,
            title=f'{self.phase_tracker_name} - {self.name}')
        return hf

    def plot_evoked(self, filt_bandpass_hz: Optional[tuple] = None):

        from concurrent.futures import ProcessPoolExecutor

        # Use a process pool to compute the evoked responses in parallel.
        results_data = []
        with ProcessPoolExecutor() as executor:
            results_data = list(
                tqdm(executor.map(simulation_group_process_result,
                                  self.results,
                                  [filt_bandpass_hz] * len(self.results)),
                     total=len(self.results)))

        if not results_data:
            raise ValueError(
                "No valid results found to compute evoked response.")

        # Unpack the results.
        grp_mean_response = [r[0] for r in results_data]
        grp_ci = [r[1] for r in results_data]
        # Assume that all time_axis values are identical.
        time_axis = results_data[0][2]

        # plot the evoked response for each result in grey with alpha
        hf, hax = plt.subplots(figsize=(8, 4))
        for mean_response, ci in zip(grp_mean_response, grp_ci):
            hax.fill_between(
                time_axis,
                ci.confidence_interval.low,
                ci.confidence_interval.high,
                color='black',
                alpha=0.1,
            )
            hax.plot(
                time_axis,
                mean_response,
                color='black',
                alpha=0.4,
                lw=1,
            )

        # take group average of the evoked responses and plot with thick line
        grp_mean_response = np.mean(grp_mean_response, axis=0)

        hax.plot(
            time_axis,
            grp_mean_response,
            color='tab:blue',
            lw=2,
        )
        hax.set_title(
            f'Evoked Response - {self.phase_tracker_name} - {self.name}')
        hax.set_xlabel('Time (s)')
        hax.set_ylabel('Signal')
        hax.grid()

        # add a note in the top left corner inside the axis box with N
        hax.text(
            0.05,
            0.95,
            f'N={self.n}',
        )

        return hf


def plot_phase_hist_array(
    stim_phases: np.ndarray,
    target_phase: float,
    title: str,
):
    '''
    Plot a polar histogram of the phase distribution.
    Static function that acts on numpy arrays.
    Used by SimulationResult and SimulationGroupResult to make consistent plots.
    '''
    mean_phase = scipy.stats.circmean(stim_phases)
    std_phase = scipy.stats.circstd(stim_phases)

    # plot polar histogram of phase
    hf, hax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw=dict(polar=True))
    hax.hist(stim_phases,
             bins=30,
             range=(0, 2 * pi),
             color='slateblue',
             edgecolor='black',
             linewidth=0.5)
    # Maximum radius for extending lines
    r_max = 1.2 * hax.get_ylim()[1]

    # Extend beyond the edge of the circular plot
    extended_r = r_max

    # Draw line at mean
    hax.plot([mean_phase] * 2, [0, extended_r],
             color='red',
             linestyle='--',
             linewidth=1,
             label='mean phase')

    # Draw line at target
    hax.plot([target_phase] * 2, [0, extended_r],
             color='black',
             linestyle='--',
             linewidth=1,
             label='target phase')

    hax.set_xticks([0, pi / 2, pi, 3 * pi / 2])
    hax.set_xticklabels(
        ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
    hax.set_yticks(hax.get_yticks()[-1:])
    hax.grid(True, alpha=0.2)
    # hax.set_xlabel('Phase')
    hax.set_title(f'{title}\nMean: {mean_phase:.2f}, Std: {std_phase:.2f}')

    return hf


def simulation_group_process_result(result, filt_bandpass_hz: Optional[tuple]):
    # Check if there are stimulation triggers
    if len(result.stims_sp) == 0:
        print(
            f'No stimulation triggers found for {result.Dataset.name}. Skipping this result.'
        )
        return None
    # Compute the evoked responses and corresponding time axis
    evoked_responses, time_axis = result.compute_timelocked(
        filt_bandpass_hz=filt_bandpass_hz)

    # Define a statistic function for bootstrapping
    statistic = lambda data: np.mean(data, axis=0)

    # Compute the bootstrapped 95% confidence interval
    ci = scipy.stats.bootstrap(
        (evoked_responses, ),
        statistic,
        confidence_level=0.95,
        n_resamples=250,
        method='basic',
        axis=0,
    )

    # Return the mean evoked response, the CI, and the time axis
    mean_response = np.mean(evoked_responses, axis=0)
    return mean_response, ci, time_axis


def run_simulations(
    dataset: SimulationDataset,
    phase_tracker: object,
    block_size_sp: int = 12,
):
    '''
    Run simulations on a dataset using a phase tracker.
    The simulation processes the dataset in blocks of samples.
    Args:
        dataset (SimulationDataset): The dataset containing the signal.
        phase_tracker (object): The phase tracker to be used for processing.
        block_size_sp (int): Size of the blocks to process at a time.
    Returns:
        SimulationResult: A result object containing the processed dataset and phase tracker.
    '''
    signal = dataset.signal
    # Convert NaNs in the signal to zeros
    signal = np.nan_to_num(signal, nan=0.0)

    # Initialize stim_trigs
    stims_sp = []
    internals_ts = []
    status_ts = []

    # Process the signal in blocks
    for i in range(0, len(signal), block_size_sp):
        block = signal[i:i + block_size_sp]
        for si, sample in enumerate(block):
            result, internals = phase_tracker.update(sample)
            if result.status & PhaseTrackerStatus.STIM1:
                stims_sp.append(i + si + result.delay_sp)

            status_ts.append(result.status)
            internals_ts.append(internals)
    stims_sp = list(filter(lambda x: x >= 0 and x < len(signal), stims_sp))

    return SimulationResult(dataset, phase_tracker, stims_sp, status_ts,
                            internals_ts)


def generate_sine(
    fs: int = 256,
    freq: float = 1.25,
    duration: float = 30.0,
    amplitude: float = 90.0,
):
    '''
    Generate a sine wave signal with a specified frequency and duration.
    The signal is sampled at a given sampling frequency.
    Args:
        fs (int): Sampling frequency.
        freq (float): Frequency of the sine wave.
        duration (float): Duration of the signal in seconds.
    Returns:
        SimulationDataset: A dataset containing the generated signal.
    '''

    t = np.arange(0, duration, 1 / fs)
    test_signal = amplitude * np.sin(2 * pi * t * freq) 

    return SimulationDataset(
        t=t,
        signal=test_signal,
        fs=fs,
        name=f'sine_{freq}Hz',
    )


def generate_timevarying_sine(
    fs: int = 256,
    amplitude: float = 1.0,
    freq: float = 1.25,
    duration: float = 30.0,
    freq_change: float = 0.5,
):
    '''
    Generate a time-varying sine wave signal with frequency modulation.
    The frequency of the sine wave changes over time according to a sine function.
    The modulation frequency is defined by the `freq_change` parameter.

    Args:
        fs (int): Sampling frequency.
        freq (float): Base frequency of the sine wave.
        duration (float): Duration of the signal in seconds.
        freq_change (float): Frequency modulation amplitude.

    Returns:
        SimulationDataset: A dataset containing the generated signal.
    '''

    t = np.arange(0, duration, 1 / fs)
    w_n = 1 + freq_change * np.sin((2 * pi * t) / duration)
    phi = 2 * pi * np.cumsum(w_n) / fs
    test_signal = amplitude * np.sin(phi)

    return SimulationDataset(
        t=t,
        signal=test_signal,
        fs=fs,
        name=f'timevarying_sine_{freq}Hz',
    )


def add_noise(
    dataset: SimulationDataset,
    noise_level: float = 0.1,
):
    '''
    Add Gaussian noise to a signal dataset.
    Args:
        dataset (SimulationDataset): The dataset containing the original signal.
        noise_level (float): Standard deviation of the Gaussian noise.
    Returns:
        SimulationDataset: A new dataset with added noise.
    '''

    noise = np.random.normal(0, noise_level, len(dataset.signal))
    noisy_signal = dataset.signal + noise

    return SimulationDataset(
        t=dataset.t,
        signal=noisy_signal,
        fs=dataset.fs,
        name=f'{dataset.name}_noisy',
    )


def get_anphy_datasets(study: Literal['SC', 'ST'] = 'ST'):
    """""
    Identify subject IDs from the ANPHY sleep dataset

    Data should be placed in the relative directory 'data/anphy_sleep/'
    """
    import glob
    import os

    base_path = 'data/anphy_sleep'
    folders = glob.glob(os.path.join(base_path, '*/')) # Ensure we are looking for directories

    # Each folder in folders is a subject ID (folder name)
    subject_ids = [os.path.basename(os.path.normpath(path)) for path in folders if os.path.isdir(path)]

    return subject_ids

def load_anphy_data(subject: str = 'EPCTL01',
                   max_duration_s: Optional[float] = None):
    """
    Load the Anphy sleep data for a given subject.
    Data is expected to be in '.npy' format.
    The path can be specified using the ANPHY_PROCESSED_PATH environment variable,
    otherwise it defaults to 'data/anphy_processed/'.
    """
    import numpy as np
    import os

    base_path = 'data/anphy_processed'
    path = os.path.join(base_path, f'{subject}.npy')

    eeg_signal = np.load(path)[0]
    fs = 256
    if max_duration_s is not None:
        max_length = min(len(eeg_signal), int(max_duration_s * fs))
        eeg_signal = eeg_signal[:max_length]
    return SimulationDataset(
        t=np.arange(len(eeg_signal)) / fs,
        signal=eeg_signal* 1e6,  # Convert to microvolts
        fs=fs,
        name=f'{subject}_raweeg',
    )