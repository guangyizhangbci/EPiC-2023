import pandas as pd
import os 
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, lfilter, lfilter_zi, filtfilt, sosfreqz, resample
import numpy as np
import torch 

class FILTER_ECG(torch.nn.Module):
    def __init__(self):
        super(FILTER_ECG, self).__init__()
        self.fs = 1000 # sampling_rate
        self.new_fs  = 200 # Define the new sampling rate for downsampling
        self.lowcut  = 1
        self.highcut = 30
        self.order   = 20


    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos,
                    data)  # Filter data along one dimension using cascaded second-order sections. Using lfilter for each second-order section.
        return y

    def butter_bandpass_filter_once(self, data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        # Apply the filter to data. Use lfilter_zi to choose the initial condition of the filter.
        zi = sosfilt_zi(sos)
        z, _ = sosfilt(sos, data, zi=zi * data[0])
        return sos, z, zi


    def butter_bandpass_filter_again(self, sos, z, zi):
        # Apply the filter again, to have a result filtered at an order the same as filtfilt.
        z2, _ = sosfilt(sos, z, zi=zi * z[0])
        return z2


    def butter_bandpass_forward_backward_filter(self, data, lowcut, highcut, fs, order=1):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos,
                        data)  # Apply a digital filter forward and backward to a signal.This function applies a linear digital filter twice, once forward and once backwards. The combined filter has zero phase and a filter order twice that of the original.
        return y


    def forward(self, ecg):

        new_length = int(ecg.shape[0] * self.new_fs / self.fs)

        # Downsample the ECG signal
        downsampled_ecg_signal = signal.resample(ecg, new_length)

        ecg_filtered = self.butter_bandpass_filter(downsampled_ecg_signal, self.lowcut, self.highcut, self.new_fs, order=self.order)

        # plt.plot(ecg_filtered)
        # plt.show()
        # exit(0)


        wavelet = pywt.Wavelet('db4')
        levels = 4
        coeffs = pywt.wavedec(ecg_filtered, wavelet, level=levels)
        # threshold = np.sqrt(2 * np.log(ecg_filtered.size))
        threshold = 0.1 * np.max(coeffs[-1])
        coeffs_filtered = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # Reconstruct filtered signal
        ecg_denoised = pywt.waverec(coeffs_filtered, wavelet)


        return ecg_denoised




class FILTER_GSR(torch.nn.Module):
    def __init__(self):
        super(FILTER_GSR, self).__init__()
        self.fs = 1000 # sampling_rate
        self.new_fs  = 200 # Define the new sampling rate for downsampling
        self.lowcut  = 0.05
        self.highcut = 5
        self.order   = 4


    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos,
                    data)  # Filter data along one dimension using cascaded second-order sections. Using lfilter for each second-order section.
        return y



    def forward(self, gsr):

        new_length = int(gsr.shape[0] * self.new_fs / self.fs)

        # Downsample the ECG signal
        downsampled_gsr_signal = signal.resample(gsr, new_length)

        plt.plot(downsampled_gsr_signal)
        downsampled_gsr_signal = downsampled_gsr_signal - np.mean(downsampled_gsr_signal)
        gsr_filtered = self.butter_bandpass_filter(downsampled_gsr_signal, self.lowcut, self.highcut, self.new_fs, order=self.order)

        # plt.plot(gsr_filtered)
        # plt.show()
        # exit(0)
        return gsr_filtered













