import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import os

# Note: This file is refactored from the original element_extraction-integrate.py script.

def detect_calcium_transients(data, fs=4.8, params=None):
    """
    Detects calcium transients in calcium imaging data.
    All parameters are passed via the 'params' dictionary.
    """
    default_params = {
        'min_snr': 3.5, 'min_duration': 12, 'smooth_window': 31, 'peak_distance': 24,
        'baseline_percentile': 8, 'max_duration': 800, 'detect_subpeaks': False,
        'subpeak_prominence': 0.15, 'subpeak_width': 5, 'subpeak_distance': 8,
        'min_morphology_score': 0.20, 'min_exp_decay_score': 0.12, 'filter_strength': 1.0
    }
    if params is None:
        params = {}
    
    run_params = default_params.copy()
    run_params.update(params)

    # Use local variables for easier access
    p = run_params
    min_snr, min_duration, smooth_window, peak_distance = p['min_snr'], p['min_duration'], p['smooth_window'], p['peak_distance']
    baseline_percentile, max_duration, filter_strength = p['baseline_percentile'], p['max_duration'], p['filter_strength']
    min_morphology_score, min_exp_decay_score = p['min_morphology_score'], p['min_exp_decay_score']
    detect_subpeaks, subpeak_prominence, subpeak_width, subpeak_distance = p['detect_subpeaks'], p['subpeak_prominence'], p['subpeak_width'], p['subpeak_distance']

    if filter_strength != 1.0:
        min_snr *= filter_strength
        min_morphology_score = min(0.9, min_morphology_score * filter_strength)
        min_exp_decay_score = min(0.9, min_exp_decay_score * filter_strength)
        if filter_strength > 1.0:
            min_duration = int(min_duration * (1 + (filter_strength - 1) * 0.5))
            smooth_window = int(smooth_window * (1 + (filter_strength - 1) * 0.3))
            peak_distance = int(peak_distance * (1 + (filter_strength - 1) * 0.2))
        else:
            min_duration = max(10, int(min_duration * filter_strength))
            smooth_window = max(21, int(smooth_window * (1 - (1 - filter_strength) * 0.3)))
            peak_distance = max(10, int(peak_distance * filter_strength))
    
    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1
    smoothed_data = signal.savgol_filter(data, smooth_window, 3) if smooth_window > 1 else data.copy()
    
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    if noise_level == 0:
        noise_level = 1e-9 # Avoid division by zero
        
    threshold = baseline + min_snr * noise_level
    
    prominence_threshold = noise_level * 1.2 * filter_strength
    min_width_frames = min_duration // 6
    
    peaks, peak_props = find_peaks(smoothed_data, height=threshold, prominence=prominence_threshold, width=min_width_frames, distance=peak_distance)
    
    if len(peaks) == 0:
        return [], smoothed_data
        
    widths_info = peak_widths(smoothed_data, peaks, rel_height=0.5)
    fwhms = widths_info[0]

    transients = []
    for i, peak in enumerate(peaks):
        start_time = int(widths_info[2][i])
        end_time = int(widths_info[3][i])
        duration_frames = end_time - start_time
        
        if not (min_duration <= duration_frames <= max_duration):
            continue
        
        amplitude = smoothed_data[peak] - baseline
        
        transients.append({
            'start': start_time, 
            'peak': peak, 
            'end': end_time, 
            'amplitude': amplitude,
            'duration': duration_frames / fs,
            'fwhm': fwhms[i] / fs,
            'rise_time': (peak - start_time) / fs,
            'decay_time': (end_time - peak) / fs,
            'auc': trapezoid(smoothed_data[start_time:end_time] - baseline, dx=1/fs),
            'snr': amplitude / noise_level
        })
        
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=4.8, visualize=False, params=None):
    if params is None:
        params = {}
    transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs, params=params)
    if not transients:
        return pd.DataFrame(), None, smoothed_data
    feature_table = pd.DataFrame(transients)
    if visualize:
        fig = visualize_calcium_transients(neuron_data, smoothed_data, transients, fs=fs)
        return feature_table, fig, smoothed_data
    return feature_table, None, smoothed_data

def visualize_calcium_transients(raw_data, smoothed_data, transients, fs=4.8):
    fig, ax = plt.subplots(figsize=(20, 6))
    time_axis = np.arange(len(raw_data)) / fs
    ax.plot(time_axis, raw_data, color='grey', alpha=0.6, label='Raw Signal')
    ax.plot(time_axis, smoothed_data, color='blue', label='Smoothed Signal')
    for transient in transients:
        start_time, peak_time, end_time = transient['start']/fs, transient['peak']/fs, transient['end']/fs
        ax.axvspan(start_time, end_time, color='yellow', alpha=0.3)
        ax.plot(peak_time, smoothed_data[transient['peak']], 'rv')
    ax.set_title("Detected Calcium Transients")
    ax.set_xlabel(f"Time (seconds, fs={fs}Hz)")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.legend()
    return fig

def analyze_all_neurons_transients(data_df, neuron_columns, fs=4.8, start_id=1, file_info=None, params=None):
    all_transients = []
    for neuron in neuron_columns:
        feature_table, _, _ = extract_calcium_features(data_df[neuron].values, fs=fs, params=params)
        if not feature_table.empty:
            feature_table['neuron_id'] = neuron
            feature_table['event_id'] = range(start_id, start_id + len(feature_table))
            start_id += len(feature_table)
            all_transients.append(feature_table)
    if not all_transients:
        return pd.DataFrame(), start_id
    final_table = pd.concat(all_transients, ignore_index=True)
    if file_info:
        for key, value in file_info.items():
            final_table[key] = value
    return final_table, start_id

def run_batch_extraction(file_paths, output_dir, fs=4.8, **kwargs):
    all_results = []
    current_event_id = 1
    for file_path in file_paths:
        try:
            data_df = pd.read_excel(file_path, sheet_name='dF', header=0)
            neuron_columns = data_df.columns[1:]
            file_info = {'source_file': os.path.basename(file_path)}
            result_df, next_start_id = analyze_all_neurons_transients(
                data_df=data_df, neuron_columns=neuron_columns, fs=fs,
                start_id=current_event_id, file_info=file_info, params=kwargs
            )
            if not result_df.empty:
                all_results.append(result_df)
                current_event_id = next_start_id
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            continue
    if not all_results:
        return None
    final_df = pd.concat(all_results, ignore_index=True)
    output_filename = f"batch_run_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}_features.xlsx"
    output_path = os.path.join(output_dir, output_filename)
    final_df.to_excel(output_path, index=False)
    return output_path