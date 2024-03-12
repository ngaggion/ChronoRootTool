import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2
import csv
import re
import pandas as pd
import os
from .report import load_path as loadPath

def recorrerRaiz(raiz, LR = False):
    if LR:
        raiz = raiz[0]

    p = []
    for punto in raiz[0]:
        x = int(punto.attrib['x'])
        y = int(punto.attrib['y'])
        p.append([x,y])
  
    ini = p[0]
    fin = p[-1]

    if ini[1] > fin[1]:
        return p[::-1]
    else:
        return p


def tipAngle(puntos):
    ini = puntos[0]
    fin = puntos[-1]
    hipot = np.sqrt((ini[0]-fin[0])**2 + (ini[1]-fin[1])**2)
    cat = fin[1]-ini[1]

    angle_rad = np.arccos(cat/hipot)
    angle_deg = angle_rad*180/np.pi

    return angle_deg

def emergenceAngle(puntos, largo = 1, pixelsize = 0.04):            
    l = int(largo/pixelsize)
    N = len(puntos)
    ini = puntos[0]
    fin = puntos[min(l, N)-1]

    hipot = np.sqrt((ini[0]-fin[0])**2 + (ini[1]-fin[1])**2)
    cat = fin[1]-ini[1]

    angle_rad = np.arccos(cat/hipot)
    angle_deg = angle_rad*180/np.pi

    return angle_deg


def plotAngles(ax, puntos, angle, tip = True, largo = 1, pixelsize = 0.04):
    if tip:
        ini = puntos[0]
        fin = puntos[-1]
    else:
        l = int(largo/pixelsize)
        N = len(puntos)

        ini = puntos[0]
        fin = puntos[min(l, N)-1]

    xs = ini[0], fin[0]
    ys = ini[1], fin[1]
    ax.plot(xs, ys, linewidth = 0.5)

    xs = ini[0], fin[0]
    ys = ini[1], fin[1]
    ax.plot([ini[0], ini[0]], ys, linewidth = 0.5)

    ax.text(x = ini[0], y = ini[1], s = str(round(angle,2)),fontsize = 'xx-small')

    return


def find_nearest(point, listpoints):
    d = np.linalg.norm(point-listpoints, axis = 1)
    p = np.argmin(d)

    return p, d[p]

def matching(newInis, allNewPoints, oldInis, allOldPoints, oldNames, NRoots = 0):
    # newInis : begin of lateral roots - this run
    # allNewPoints : all points of the lateral root polyline - this run
    
    # oldInis : begin of lateral roots
    # allOldPoints : all points of the lateral root polyline
    # oldNames : names in previous iteration

    matchedNames = []
    matchedInis = []
    matchedPoints = []

    seen = []

    if oldInis == [] or oldNames == []:
        for j in range(0,len(newInis)):
            matchedNames.append('LR%s'%NRoots)
            matchedInis.append(newInis[j])
            matchedPoints.append(allNewPoints[j])
            NRoots += 1
    else:
        op = np.array(oldInis)
        nps = np.array(newInis)

        for j in range(0, nps.shape[0]):
            p, dp = find_nearest(nps[j,:], op)

            if dp < 20:
                matchedNames.append(oldNames[p])
                matchedInis.append(newInis[j])
                matchedPoints.append(allNewPoints[j])
                seen.append(p)
            else:
                matchedNames.append('LR%s'%NRoots)
                matchedInis.append(newInis[j])
                matchedPoints.append(allNewPoints[j])
                NRoots += 1
        
        N_old = op.shape[0]
        seen.sort()

        for j in range(0, N_old):
            if j not in seen:
                matchedNames.append(oldNames[j])
                matchedInis.append(oldInis[j])
                matchedPoints.append(allOldPoints[j])

        idxs = np.argsort(np.array(matchedNames))
        matchedNames = np.array(matchedNames,dtype=object)[idxs].tolist()
        matchedInis = np.array(matchedInis,dtype=object)[idxs].tolist()
        matchedPoints = np.array(matchedPoints,dtype=object)[idxs].tolist()   

    return matchedInis, matchedPoints, matchedNames, NRoots

def lenRoot(points, pixel_size = 0.04):
    accum = 0
    for j in range(1, len(points)):
        d = np.linalg.norm(np.array(points[j]) - np.array(points[j-1]))
        accum += (d * pixel_size)
    
    return accum


def getAngles(conf, path):
    paths = loadPath(os.path.join(path,'RSML'), '*')

    LateralRoots = []
    LateralRootsInis = []
    LateralRootsNames = []
    NRoots = 0
    
    filepath = os.path.join(path, 'LateralRootsData.csv')
    
    with open(filepath, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(["Img", "Number of lateral roots", "Mean tip angle", "Mean emergence angle", 
        "First LR tip", "First LR emergence"])

        for step in paths:
            arbol = ET.parse(step).getroot()

            with open(os.path.join(path, 'metadata.json')) as f:
                metadata = json.load(f)

            y1,y2,x1,x2 = metadata['bounding box']
            h = y2-y1
            w = x2-x1

            plant = arbol[1][0][0]
            
            if len(plant) > 1:
                LRs = plant[1:]
                
                pts = []
                inis = []

                ## Primero hago el tracking
                for LR in LRs:
                    puntos = recorrerRaiz(LR, True)
                    pts.append(puntos)
                    inis.append(puntos[0])

                LateralRootsInis, LateralRoots, LateralRootsNames, NRoots = matching(inis, pts, LateralRootsInis, LateralRoots, LateralRootsNames, NRoots)
                
                ## Luego estimo los angulos
                tips = []
                emergs = []
                lengths = []

                for root in LateralRoots:
                    TipAngle = tipAngle(root)
                    EmergenceAngle = emergenceAngle(root, float(conf['emergenceDistance']), metadata['pixel_size'])

                    tips.append(TipAngle)
                    emergs.append(EmergenceAngle)

            imgname = step.replace(os.path.join(path,'RSML')+'/', '').replace('.rsml','.png')
            
            if NRoots == 0:
                measures = [imgname, 0, 0, 0, 0, 0]
            else:
                measures = [imgname, NRoots, round(np.mean(tips), 3), round(np.mean(emergs), 3), round(tips[0], 3), round(emergs[0], 3)]
                    
            writer.writerow(measures)
    
    return


def dataWork(data, d0, h0, dend, hend):
    for i in range(0, data.shape[0]):
        name = data.loc[i, "Img"]
        nums = re.findall(r'\d+', name)
        dia = int(nums[2])
        hora = int(nums[3])
        minutos = int(nums[4])

        data.loc[i, "Day"] = dia
        data.loc[i, "Hour"] = hora
        data.loc[i, "Minutes"] = minutos
    
    data = data.drop(['Img'], axis=1)
    data2 = pd.DataFrame(columns = data.columns[:-1])
    
    d1 = data['Day'][0]
    h1 = data['Hour'][0]
    
    for dia in range(int(d0), int(d1)):
        if dia == d0:
            h_ini = int(h0)
        else:
            h_ini = 0

        for hora in range(h_ini, 24):
            aux = pd.DataFrame([[0,0,0,0,0, dia, hora]], columns = data2.columns)
            data2 = pd.concat([data2, aux], ignore_index=True)
    
    for hora in range(0, int(h1)):
        aux = pd.DataFrame([[0,0,0,0,0, d1, hora]], columns = data2.columns)
        data2 = pd.concat([data2, aux], ignore_index=True)
    
    if dend < d1:
        dend = dend + data['Day'].max()

        data['Day'] = data['Day'].apply(lambda x: x + data['Day'].max() if x < d1 else x)

    for dia in range(int(d1), int(dend)+1):
        if dia == d1:
            h_ini = int(h1)
        else:
            h_ini = 0
        if dia == dend:
            h_end = hend + 1
        else:
            h_end = 24

        for hora in range(h_ini, h_end):
            aux = data[data['Day'] == dia]
            aux = aux[aux['Hour'] == hora]
            aux = aux.mean(0)[:-1]
            
            if aux.shape[0] == 10:
                sub = data.iloc[-1, :].copy()
                sub.iloc[-2] = dia
                sub.iloc[-1] = hora
                aux = sub
            
            # aux values to list
            aux = aux.tolist()[:5] + [dia, hora]
            aux = pd.DataFrame([aux], columns = data2.columns)
            data2 = pd.concat([data2, aux], ignore_index=True)

    data2['i'] = data2.index
    data2['Day'] -= d0
    
    data2 = data2.astype('float')
    
    print(data2.colums)

    return data2

import seaborn as sns
import matplotlib.pyplot as plt

def avoidIncreasingValues(data, metric, tol = 0.3):
    # first perform a median filter to avoid spikes
    data[metric] = data[metric].rolling(window=5, min_periods=1).median()
    
    # then check if there no high jumps given a tolerance of 30%
    series = data[metric]
    for j in range(12, len(series)):        
        if series.iloc[j] < series.iloc[j-1] * (1+tol):
            continue
        else:
            series.iloc[j] = series.iloc[j-1]
    data[metric] = series
    
    return data

def getFirstLateralRoots(conf, df):
    reportPath = os.path.join(conf['MainFolder'], 'Report')
    
    plantas = df['Plant_id'].unique()
    
    print("Synchronizing 1st LR tip")
    
    # Add a column to the dataframe to store if the datapoint is original or extended
    df['Real'] = 1

    s = 72
    syncro = pd.DataFrame()

    for planta in plantas:
        p = df[df['Plant_id']==str(planta)]
        p = p[p['First LR tip'] > 0]
        k = p.shape[0]
        
        if k > 0:
            p = avoidIncreasingValues(p, 'First LR tip')
            
            while k < s:
                aux = p.iloc[-1, :].copy()
                aux = aux.to_frame().T
                if aux.shape[0] != 1 or aux.shape[1] != 11:
                    print(p.shape, aux.shape)
                    raise ValueError('Extending 1st LR Error')
                aux['Real'] = 0
                p = pd.concat([p, aux], ignore_index=True)
                k = k+1
            else:
                p = p.iloc[:72, :]
            
            p = p.reset_index()
            p['Time'] = p.index
            p = p.drop(['index'], axis = 1)
            p = p.loc[:,['Time', 'First LR tip', 'Experiment', 'Plant_id', 'Real']]
            
            syncro = pd.concat([syncro, p], ignore_index = True)

    syncro.to_csv(os.path.join(reportPath, 'SyncronizedFirstLR.csv'), index=False)
    
    return syncro

def makeLateralAnglesPlots(conf):
    parent_folder = conf['MainFolder']
    analysis = os.path.join(conf['MainFolder'], "Analysis")
    experiments = loadPath(analysis, '*')
    limit = conf['Limit']
    reportPath = os.path.join(parent_folder, 'Report')
    reportPath_angle = os.path.join(reportPath, 'Angles Analysis')
    os.makedirs(reportPath_angle, exist_ok=True)
    
    all_data = pd.DataFrame()
    
    print('Angles report')
    
    if os.path.exists(os.path.join(reportPath, 'LateralRootsData.csv')):
        all_data = pd.read_csv(os.path.join(reportPath, 'LateralRootsData.csv'))
        all_data['Experiment'] = all_data['Experiment'].astype('str')
        all_data['Plant_id'] = all_data['Plant_id'].astype('str')
    else:
        for exp in experiments:
            plants = loadPath(exp, '*/*/*')
            exp_name = exp.replace(analysis, '').replace('/','')
            print('Experiment:', exp_name, '- Total plants', len(plants))

            for plant in plants:
                results = loadPath(plant, '*')
                if len(results) == 0:
                    continue
                else:
                    results = results[-1]
                plant_name = plant.replace(exp, '').replace('/','_')

                file = os.path.join(results, 'LateralRootsData.csv')
                file2 = os.path.join(results, 'PostProcess_Original.csv')

                data2 = pd.read_csv(file2)
                data2.dropna(inplace=True)
                
                date1 = data2.loc[0, "Date"]
                # using pandas extract day and hour from date1
                date1 = pd.to_datetime(date1)
                d0, h0 = date1.day, date1.hour

                date2 = data2.loc[data2.shape[0]-1, "Date"]
                # using pandas extract day and hour from date2
                date2 = pd.to_datetime(date2)
                dend, hend = date2.day, date2.hour
                            
                data = pd.read_csv(file)
                data = dataWork(data, d0, h0, dend, hend)

                data['Plant_id'] = plant_name
                data['Experiment'] = exp_name

                all_data = pd.concat([all_data, data], ignore_index=True)

        # save
        all_data.to_csv(os.path.join(reportPath, 'LateralRootsData.csv'), index=False)

    frame = []
    for day in conf['daysAngles'].split(','):
        aux = all_data[all_data['Day'] == int(day)]
        aux = aux[aux['Hour'] == 0]
        aux = aux[aux['Mean emergence angle'] > 0]

        frame.append(aux)
    
    frame = pd.concat(frame)

    n_exp = len(frame['Experiment'].unique())

    plt.figure(figsize = (8, 9), dpi = 200)
    sns.color_palette("tab10")
                      
    ax = plt.subplots()

    sns.violinplot(x = 'Day', y = 'Mean emergence angle', data=frame, hue = 'Experiment', inner=None, 
                        showmeans=True, zorder=2, legend = False)
    ax = sns.swarmplot(x = 'Day', y = 'Mean emergence angle', data=frame, hue = 'Experiment', dodge= True, 
                       size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2)
    ax.set_ylim(-20, 120)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_exp], labels[0:n_exp], loc=4)

    ax.set_title('Mean emergence angle')

    plt.savefig(os.path.join(reportPath_angle, 'Mean Emergence Angle.png'), dpi = 300)
    plt.savefig(os.path.join(reportPath_angle, 'Mean Emergence Angle.svg'), dpi = 300)
    
    performStatisticalAnalysisAngles(conf, frame, 'Mean emergence angle')

    syncro = getFirstLateralRoots(conf, all_data)
    
    plt.figure(figsize = (6, 4), dpi = 200)

    sns.lineplot(y = "First LR tip", x = 'Time', hue = "Experiment", data = syncro, errorbar='se')
    plt.title('Decay of the tip angle (1st LR)')
    plt.xlabel('Time (h)')
    plt.ylabel('Angle')

    plt.savefig(os.path.join(reportPath_angle, 'First LR Tip Angle Decay.png'), dpi = 200)
    plt.savefig(os.path.join(reportPath_angle, 'First LR Tip Angle Decay.svg'), dpi = 200)
    
    plt.figure(figsize = (6, 4), dpi = 200)
    sns.lineplot(y = "Real", x = 'Time', hue = "Experiment", data = syncro, errorbar=None, estimator=np.count_nonzero)
    plt.title('Number of plant with first LR roots per experiment')
    plt.ylabel('Number of first LR')
    plt.xlabel('Time elapsed since emergence (h)')
    
    plt.savefig(os.path.join(reportPath_angle, 'First LR Number.png'), dpi = 200)
    plt.savefig(os.path.join(reportPath_angle, 'First LR Number.svg'), dpi = 200)
    
    performStatisticalAnalysisFirstLR(conf, syncro, 'First LR tip')
    
import scipy.stats as stats

def performStatisticalAnalysisAngles(conf, data, metric):
    data['Experiment'] = data['Experiment'].astype(str)
    data['Day'] = data['Day'].astype(int).astype(str)
        
    UniqueExperiments = data['Experiment'].unique()
    N_exp = int(len(UniqueExperiments))
    
    days = conf['daysAngles'].split(',')
    
    # Create a text file to store the results
    reportPath = os.path.join(conf['MainFolder'], 'Report', 'Angles Analysis')
    
    if "/" in metric:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric.replace('/',' over '))
    else:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric)
        
    # First row should say "Using Mann Whitney U test to compare the growth speed of different experiments"
    with open(reportPath_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        
        for day in days:                       
            # Compare every pair of experiments with Mann-Whitney U test
            f.write('Day: ' + str(day) + '\n')
            subdata = data[data['Day'] == str(day)]

            
            for i in range(0, N_exp-1):
                for j in range(i+1, N_exp):
                    exp1 = subdata[subdata['Experiment'] == UniqueExperiments[i]][metric]
                    exp2 = subdata[subdata['Experiment'] == UniqueExperiments[j]][metric]
                    
                    # Perform Mann-Whitney U test
                    try:
                        if len(exp1) == 0 or len(exp2) == 0:
                            raise Exception()
                            
                        U, p = stats.mannwhitneyu(exp1, exp2)
                        p = round(p, 6)
                        
                        # Write the mean value of each experiment
                        f.write('Mean ' + UniqueExperiments[i] + ': ' + str(round(exp1.mean(), 2)) + '\n')
                        f.write('Mean ' + UniqueExperiments[j] + ': ' + str(round(exp2.mean(), 2)) + '\n')
                        
                        # Compare the p-value with the significance level
                        if p < 0.05:
                            f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' are significantly different. P-value: ' + str(p) + '\n')
                        else:
                            f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' are not significantly different. P-value: ' + str(p) + '\n')
                    except:
                        f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' could not be compared\n')
                        
            f.write('\n')            
    return


def performStatisticalAnalysisFirstLR(conf, data, metric):
    data['Experiment'] = data['Experiment'].astype(str)    
    UniqueExperiments = data['Experiment'].unique()
    N_exp = int(len(UniqueExperiments))
        
    # Create a text file to store the results
    reportPath = os.path.join(conf['MainFolder'],'Report', 'Angles Analysis')
    
    if "/" in metric:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric.replace('/',' over '))
    else:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric)
        
    # First row should say "Using Mann Whitney U test to compare the growth speed of different experiments"
    with open(reportPath_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        f.write('Statistical analysis is performed every 6 hours\n')
        
        for hour in range(0, 70, 6):                       
            # Compare every pair of experiments with Mann-Whitney U test
            end = hour + 6
            
            f.write('Hour: ' + str(hour) + '-' + str(end) + '\n')
            
            hours = np.arange(hour, end)
            subdata = data[data['Time'].isin(hours)]

            for i in range(0, N_exp-1):
                for j in range(i+1, N_exp):
                    exp1 = subdata[subdata['Experiment'] == UniqueExperiments[i]][metric]
                    exp2 = subdata[subdata['Experiment'] == UniqueExperiments[j]][metric]
                    
                    # Perform Mann-Whitney U test
                    try:
                        if len(exp1) == 0 or len(exp2) == 0:
                            raise Exception()
                            
                        U, p = stats.mannwhitneyu(exp1, exp2)
                        p = round(p, 6)
                        
                        # Write the mean value of each experiment
                        f.write('Mean ' + UniqueExperiments[i] + ': ' + str(round(exp1.mean(), 2)) + '\n')
                        f.write('Mean ' + UniqueExperiments[j] + ': ' + str(round(exp2.mean(), 2)) + '\n')
                        
                        # Compare the p-value with the significance level
                        if p < 0.05:
                            f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' are significantly different. P-value: ' + str(p) + '\n')
                        else:
                            f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' are not significantly different. P-value: ' + str(p) + '\n')
                    except:
                        f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' could not be compared\n')
                        
            f.write('\n')            
    return