from .report import load_path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy import signal

def growthSpeedsSyncro(frame, N0 = None, N = None, normalize = False, detrend = False, root = 'MainRootLengthGrad (mm/h)', medfilt=False):
    grads_post = []
    
    for i in range(0, len(frame)):
        data = pd.read_csv(frame[i])

        data['Plant'] = i
        time = data['ElapsedTime (h)'].to_numpy().astype('int')
        newDay = data['NewDay'].to_numpy()

        if N0 is None:
            N0 = 0
        if N is None:
            N = len(time)
                
        # remove the first measurements before the new day
        if newDay[N0] != 0:
            begin = np.where(newDay == 0)[0][0]
            if begin > N0:
                N0 = begin
                N = N + begin
            else:
                N0 = N0 + (24 - time[N0])
                N = N + (24 - time[N0])
        
        N = min(N, len(time))
        N = N - N%24

        time = time[N0:N] - N0
        newDay = newDay[N0:N]
        
        mSpeed = data[root].to_numpy()
        
        where_are_NaNs = np.isnan(mSpeed)
        mSpeed[where_are_NaNs] = 0.0

        if normalize:
            mean = np.mean(mSpeed)
            std = np.std(mSpeed)
            if std != 0:
                mSpeed = (mSpeed - mean) / std
            else:
                mSpeed -= mean
            
        where_are_NaNs = np.isnan(mSpeed)
        mSpeed[where_are_NaNs] = 0.0

        if medfilt:
            mSpeed = signal.medfilt(mSpeed, 5)
            mSpeed = mSpeed - signal.medfilt(mSpeed, 25)
        
        if detrend:
            mSpeed = signal.detrend(mSpeed)

        grads_post.append(mSpeed[N0:N])
        
    return grads_post, time, newDay


def process(signal, time, N0, N):
    n = len(signal)
    
    timeb = time - time[0]
    deltat = 1
    deltaf = 1 / (deltat * (N-N0))

    i = 0
    s = signal[i]
    fft = np.abs(np.fft.fft(s))
    fourier = fft
    aux = timeb * deltaf
    
    dataframe = pd.DataFrame(data = {'Time': time, 'Freqs': timeb*deltaf, 'i' : i, 'Signal' : s, 'FFT' : fft})
    
    for i in range(1, n):
        s = signal[i]
        fft = np.abs(np.fft.fft(s))
        fourier = fourier + fft
        aux = pd.DataFrame(data = {'Time': time, 'Freqs': timeb*deltaf, 'i' : i, 'Signal' : s, 'FFT' : fft})
        dataframe = pd.concat([dataframe, aux])
        
    fourier = fourier / n
    
    return dataframe, fourier


def readData(experiments, normalize = False, detrend = False, root = 'MainRootLengthGrad (mm/h)', medfilt=False):
    dfs = []
    fouriers = []

    for exp in experiments:
        plants = load_path(exp, '*/*/*')
        speeds = []

        for plant in plants:
            results = load_path(plant, '*')
            if results == []:
                continue
            else:
                results = results[-1]
            speeds.append(os.path.join(results, "PostProcess_Hour.csv"))

        signal1, time, v = growthSpeedsSyncro(speeds, normalize = normalize, detrend = detrend, root = root, medfilt=medfilt)
        N0 = 0
        N = len(time)
        df, fourier = process(signal1, time, N0, N)
        df['Type'] = exp.split('/')[-1]

        dfs.append(df)
        fouriers.append(fourier)

    return pd.concat(dfs, ignore_index=True), fouriers, time 

def makeFourierPlots(conf):
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')
    
    reportPath = os.path.join(conf['MainFolder'],'Report')
    # Fourier analysis of the growth
    reportPath_fourier = os.path.join(reportPath, 'Fourier')
    
    if not os.path.exists(reportPath_fourier):
        os.makedirs(reportPath_fourier)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig3 = plt.figure(figsize=(12,12), constrained_layout=True)
    gs = fig3.add_gridspec(2, 2)
    f_ax1 = fig3.add_subplot(gs[0, 0])
    f_ax2 = fig3.add_subplot(gs[0, 1])
    f_ax3 = fig3.add_subplot(gs[1, 0])
    f_ax4 = fig3.add_subplot(gs[1, 1])

    all_frames, fouriers, time = readData(experiments)

    sns.lineplot(x="Time", y = "Signal", data = all_frames, hue="Type", errorbar='sd', ax=f_ax1, estimator=np.median)
    for j in range(0, len(time)):
        if j % 24 == 0:
            f_ax1.axvline(j, color = 'green')
            
    f_ax1.set_ylabel('Speed (mm/h)')
    f_ax1.set_xlabel('Time (h)')
    f_ax1.set_title('MR Growth Speed', fontsize = 16)
    handles, labels = ax=f_ax1.get_legend_handles_labels()
    f_ax1.legend(handles, labels, loc=2)

    all_frames, _, _ = readData(experiments, root="TotalLengthGrad (mm/h)")

    sns.lineplot(x="Time", y = "Signal", data = all_frames, hue="Type", errorbar='sd', ax=f_ax2, estimator=np.median)
    for j in range(0, len(time)):
        if j % 24 == 0:
            f_ax2.axvline(j, color = 'green')
            
    f_ax2.set_ylabel('Speed (mm/h)')
    f_ax2.set_xlabel('Time (h)')
    f_ax2.set_title('TR Growth Speed', fontsize = 16)
    handles, labels = ax=f_ax2.get_legend_handles_labels()
    f_ax2.legend(handles, labels, loc=2)

    all_frames, fouriers, _ = readData(experiments, normalize = True, medfilt = True)

    timeb = time - time[0]
    
    deltat = 1
    deltaf = 1 / (deltat * len(time))

    peak_12 = 0
    peak_24 = 0

    for i in range(0, len(timeb)):
        freq = timeb[i]*deltaf
        
        if np.abs(freq - 1/24) < 0.0001:
            peak_24 = fouriers[0][i]
            
        if np.abs(freq - 1/12) < 0.0001:
            peak_12 = fouriers[0][i]

    sns.lineplot(x = 'Freqs', y = 'FFT', hue = 'Type', data = all_frames, errorbar = 'sd', ax=f_ax3)

    f_ax3.axvline(x = 1/24, ymin = 0, ymax = peak_24/25, color = 'red')
    f_ax3.axvline(x = 1/12, ymin = 0, ymax = peak_12/25, color = 'black')

    f_ax3.set_xlim(0, 0.5)
    f_ax3.set_ylim(0, 25)

    f_ax3.set_title('Fourier Transform', fontsize = 16)
    f_ax3.set_xlabel('Frequency (1/hour)')
    f_ax3.set_ylabel('Energy')

    exp = 2 + 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
    exp2 = 1.5 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)

    sns.lineplot(x="Time", y = "Signal", data = all_frames, hue="Type", errorbar='sd', ax=f_ax4)
    for j in range(0, len(time)):
        if j % 24 == 0:
            f_ax4.axvline(j, color = 'green')
            
    f_ax4.set_ylabel('Speed (normalized)')
    f_ax4.set_xlabel('Time (h)')
    f_ax4.set_title('MR Growth Speed', fontsize = 16)
    handles, labels = ax=f_ax4.get_legend_handles_labels()
    f_ax4.legend(handles, labels, loc=4)

    f_ax4.plot(time, exp, color = 'red')
    f_ax4.plot(time, exp2, color = 'black')

    plt.savefig(os.path.join(reportPath_fourier, "fourier.png"), dpi=200)
    