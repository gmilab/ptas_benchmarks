# Auditory Stimulation Phase-Locking Simulations

## Description

This project is designed for simulating and evaluating the performance of real-time phase estimation algorithms.

## Features

*   **Multiple Phase-Tracking Algorithms**: Includes implementations of Amplitude Thresholding, Phase-Locked Loop (PLL), Sine Fitting, Zero Crossing, and TWave. 
*   **Plotting**: Generates various plots for analysis, including time series, phase histograms, and evoked responses.
*   **Jupyter Notebook for Workflow**: An example notebook (`run_group_simulations.ipynb`) demonstrates a typical workflow for running group simulations and generating results.

## Repository Structure

*   `Algo_*.py`: Python files implementing different phase-tracking and stimulation algorithms (e.g., `Algo_PLL.py`, `Algo_TWave.py`).
*   `Simulations.py`: Core Python module containing simulation logic, data loading functions (e.g., `load_anphy_data`, `get_anphy_datasets`), and plotting utilities.
*   `Inhibitors.py`: Module defining inhibitor classes that can be used by the algorithms to control stimulation.
*   `run_group_simulations.ipynb`: An example Jupyter Notebook demonstrating how to run simulations across multiple subjects and algorithms, and how to generate group-level results and plots.

