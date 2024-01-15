""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import re
import pandas as pd
import os
import json
from analysis.report import load_path, plot_individual_plant, mkdir, plot_info_all, get_atlases, plot_atlases, plot_convex_hull
from analysis.fourier_analysis import makeFourierPlots
from analysis.lateral_angles import getAngles, makeLateralAnglesPlots

if __name__ == "__main__":
    conf = json.load(open('config.json'))
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')

    reportPath = os.path.join(conf['MainFolder'],'Report')
    if not os.path.exists(reportPath):
        os.makedirs(reportPath)
    
    all_data = pd.DataFrame()
    convex_hull = pd.DataFrame()

    print("Report generation")

    for exp in experiments:
        exp_name = exp.replace(analysis, '').replace('/','')
        print('Experiment:', exp_name)

        exp_path = os.path.join(reportPath, exp_name)
        mkdir(exp_path)

        iplots = os.path.join(exp_path, 'iplots')
        mkdir(iplots)

        rpi = load_path(exp, '*')
        for rpi in rpi:
            rpi_name = rpi.replace(exp, '').replace('/','')

            cam = load_path(rpi, '*')
            for cam in cam:
                cam_name = cam.replace(rpi, '').replace('/','')

                plants = load_path(cam, '*')
                for plant in plants:
                    plant_name = plant.replace(cam, '').replace('/','')

                    results = load_path(plant, '*')
                    
                    if len(results) == 0:
                        continue
                    else:
                        results = results[-1]
                        
                    name = rpi_name + '_' + cam_name + '_' + plant_name

                    file = os.path.join(results,'PostProcess_Hour.csv')
                    data = pd.read_csv(file)

                    data['Plant_id'] = name
                    data['Experiment'] = exp_name

                    all_data = pd.concat([all_data, data], ignore_index=True)
                    plot_individual_plant(iplots, data, name)

                    if conf['doLateralAngles']:
                        getAngles(conf, results)

        if conf['doConvex']:
            # Convex hull experiment
            reportPath_convex = os.path.join(reportPath, 'ConvexHull')
            if not os.path.exists(reportPath_convex):
                os.makedirs(reportPath_convex)

            days = conf['daysConvexHull'].split(',')
            atlases, convexDF = get_atlases(exp, days = days, rotateRoot = True)
            convexDF['Experiment'] = exp_name

            convex_hull = pd.concat([convex_hull, convexDF], ignore_index=True)

            if conf['saveImagesConvex']:
                for i in range(0, len(days)):
                    at1, at2, at3 = atlases[i]
                    plot_atlases(at1, at2, at3, reportPath_convex, exp_name, days[i])
            else:
                at1, at2, at3 = atlases[-1]
                plot_atlases(at1, at2, at3, reportPath_convex, exp_name)
        
    # save dataframes to files
    all_data.to_csv(os.path.join(reportPath, 'all_data.csv'), index=False)
    plot_info_all(reportPath, all_data)

    if conf['doConvex']:
        convex_hull.to_csv(os.path.join(reportPath_convex, 'convex_hull.csv'), index=False)
        plot_convex_hull(reportPath_convex, convex_hull)

    if conf['doFourier']:      
        makeFourierPlots(conf)
    
    if conf['doLateralAngles']:
        makeLateralAnglesPlots(conf)

    print("Report generation finished")
    