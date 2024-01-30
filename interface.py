from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox

import subprocess
import json
import os
import pathlib
import re
import sys
import cv2

def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

class AspectRatioLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resizeEvent(self, event):
        if self.pixmap():
            pixmap = self.pixmap().scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(pixmap)
        super().resizeEvent(event)

    def set_pixmap(self, pixmap, size = None):
        if size is None:
            size = self.size() 
        scaled_pixmap = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class Ui_ChronoRootAnalysis:
    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options() | QtWidgets.QFileDialog.DontUseNativeDialog
        return QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", options=options)

    def saveFieldsIntoJson(self):
        json_path = os.path.join(os.getcwd(), "config.json")
        data = {}

        for field in [self.rpiField, self.cameraField, self.plantField, self.processingLimitField, 
                      self.processingLimitField_3, self.emergenceDistanceField, self.captureIntervalField]:
            if field.text().isdigit():
                data[field.objectName()] = int(field.text())
            
            if field.text() == "":
                data[field.objectName()] = ""
                
        data.update({field.objectName(): field.text() for field in [self.identifierField, self.videoField, self.projectField]})
        data.update({field.objectName(): field.isChecked() for field in [self.saveImagesButton, self.saveImagesConvex, 
                                                                         self.doConvex, self.doFourier, self.doLateralAngles]})
        
        data["daysConvexHull"] = self.daysConvexField.text()
        data["daysAngles"] = self.daysAnglesField.text()

        # map values for compatibility with 1_analysis.py
        data["rpi"] = data["rpiField"]
        data["cam"] = data["cameraField"]
        data["plant"] = data["plantField"]
        data["identifier"] = data["identifierField"]
        data["Images"] = data["videoField"]
        data["processingLimit"] = data["processingLimitField"]
        data["timeStep"] = data["captureIntervalField"]
        data["MainFolder"] = data["projectField"]
        data["saveImages"] = data["saveImagesButton"]
        data["emergenceDistance"] = data["emergenceDistanceField"]

        if data["processingLimit"] != "":
            data['Limit'] = int(data["processingLimit"] * 24 * 60 / int(data['timeStep']))
        else:
            data['Limit'] = 0

        with open(json_path, "w") as json_file:
            json.dump(data, json_file)

    def loadJsonIntoFields(self):
        json_path = os.path.join(os.getcwd(), "config.json")
        data = json.load(open(json_path, 'r'))

        for field in [self.rpiField, self.cameraField, self.plantField, self.processingLimitField, 
                      self.processingLimitField_3, self.emergenceDistanceField, self.captureIntervalField]:
            if field.objectName() in data:
                field.setText(str(data[field.objectName()]))

        for field in [self.identifierField, self.videoField, self.projectField]:
            if field.objectName() in data:
                field.setText(data[field.objectName()])

        for field in [self.saveImagesButton, self.saveImagesConvex, self.doConvex, self.doFourier, self.doLateralAngles]:
            if field.objectName() in data:
                field.setChecked(data[field.objectName()])

        if "daysConvexHull" in data:
            self.daysConvexField.setText(str(data["daysConvexHull"]))
        if "daysAngles" in data:
            self.daysAnglesField.setText(str(data["daysAngles"]))


    def refresh_table(self):
        # Store current sort order and column
        current_sort_order = self.table.horizontalHeader().sortIndicatorOrder()
        current_sort_column = self.table.horizontalHeader().sortIndicatorSection()

        self.table.setSortingEnabled(False)

        self.table.clearContents()
        self.table.setRowCount(0)

        # Get the data from the database
        AnalysisFolder = os.path.join(self.projectField.text(), "Analysis")
        pathlib_dir = pathlib.Path(AnalysisFolder)

        data_files = pathlib_dir.glob('*/*/*/*/*')
        data_files = [str(file) for file in data_files]
        data_files = sorted(data_files, key=lambda x: natural_keys(x))

        data = []

        self.plant_dropdown.clear()

        for file in data_files:
            rel_path = os.path.relpath(file, AnalysisFolder)
            split = rel_path.split(os.path.sep)
            variety = split[0]
            rpi = split[1]
            camera = split[2]
            plant = split[3]
            results = split[4]

            # read the error rate from the log file first line
            if os.path.exists(os.path.join(file, "log.txt")):
                with open(os.path.join(file, "log.txt"), 'r') as f:
                    date = f.readline().replace("Finish time: ", "")

                    if  "No segmentation found" in date:
                        error_rate = "No segmentation found"
                    else:
                        error_rate = float(f.readline().split(':')[-1])
                        error_rate = str(round(error_rate, 4)*100) + '%'
                status = "Finished"
            else:
                date = ""
                error_rate = ""
                status = "Not finished"

            data.append([variety, rpi, camera, plant, results, error_rate, status, date, file])

            self.plant_dropdown.addItem(file)

        self.table.setRowCount(len(data))

        for row, row_data in enumerate(data):
            for col, cell_data in enumerate(row_data[:-1]):  # Ignore the last element (path)
                item = QTableWidgetItem(str(cell_data))
                item.path = row_data[-1]  # Store the path in the item
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setSortingEnabled(True)  

        # Restore the sort order and column
        self.table.sortItems(current_sort_column, current_sort_order)

        return


    def open_selected_path(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = item.path

        # Open the directory in the file explorer
        if os.name == 'nt':  # Windows
            os.startfile(path)
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{path}"')
        else:  # Linux
            os.system(f'xdg-open "{path}"')

    def remove_selected_path(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = item.path

        # Removing the plant means moving it to a folder called "Removed"
        # This is done to avoid losing the data in case the user wants to recover it
        # Also keep the same folder structure, from the Analysis folder
        removed_path = self.projectField.text() + "/Removed"
        removed_path = os.path.join(removed_path, os.path.relpath(path, self.projectField.text() + "/Analysis"))

        if not os.path.exists(os.path.dirname(removed_path)):
            os.makedirs(os.path.dirname(removed_path))

        # Open the directory in the file explorer
        if os.name == 'nt':
            os.system(f'move "{path}" "{removed_path}"')
        elif sys.platform == 'darwin':
            os.system(f'mv "{path}" "{removed_path}"')
        else:
            os.system(f'mv "{path}" "{removed_path}"')
        
        self.refresh_table()
        
        return

    def get_image_paths(self):
        if not os.path.exists(os.path.join(self.selected_plant, "log.txt")):
            return None, None, None, None
        
        metadata = json.load(open(os.path.join(self.selected_plant, "metadata.json"), 'r'))
        bbox = metadata["bounding box"]
        overlayPath = metadata["folders"]["images"] + "/SegMulti/"

        # list all images in the folder with pathlib, then sort them
        pathlib_dir = pathlib.Path(overlayPath)
        image_files = pathlib_dir.glob('*.png')
        image_files = [str(file) for file in image_files]
        image_files = sorted(image_files, key=lambda x: natural_keys(x))

        if len(image_files) == 0:
            return None, None, None, None
        
        overlay = image_files[-1]

        variety = self.selected_plant.split(os.path.sep)[-5]
        rpi = self.selected_plant.split(os.path.sep)[-4]
        camera = self.selected_plant.split(os.path.sep)[-3]
        plant = self.selected_plant.split(os.path.sep)[-2]

        filename = variety + "_" + rpi + "_" + camera + "_" + plant + ".png"
        image2_path = os.path.join(self.selected_plant, filename)

        image1_path = metadata["ImagePath"] + '/' + overlay.split(os.path.sep)[-1]

        return image1_path, image2_path, overlay, bbox


    def update_image_labels(self):
        # Get the selected plant
        self.selected_plant = self.plant_dropdown.currentText()

        # Load the images for the selected plant and time index
        # Replace the `get_image_paths` function with your own implementation that returns the paths to the images
        image1_path, image2_path, overlay, bbox = self.get_image_paths()

        # Check if image paths exist
        if image1_path is None or not os.path.exists(image1_path):
            self.image_label1.clear()
            self.image_label1.setText("Analysis is not yet finished. \n Refresh to update")
            self.image_label1.setAlignment(QtCore.Qt.AlignCenter)
            self.image_label1.show()
        else:
            # Clear the message if the image paths exist
            self.image_label1.clear()

            # Set the images to the labels
            image = cv2.imread(image1_path)
            if image is None:
                self.image_label1.setText("Analysis is not yet finished. \n Refresh to update")
                self.image_label1.setAlignment(QtCore.Qt.AlignCenter)
                self.image_label1.show()
            else:
                image = image[bbox[0]:bbox[1], bbox[2]:bbox[3]]

                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # Check if the overlay checkbox is checked and the overlay path exists
                if self.overlay_checkbox.isChecked() and os.path.exists(overlay):
                    image_overlay = cv2.imread(overlay)
                    image_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_BGR2RGB)

                    if image.shape[0] == image_overlay.shape[0] and image.shape[1] == image_overlay.shape[1] and image.shape[2] == image_overlay.shape[2]:
                        image = cv2.add(image, image_overlay)

                height, width, channel = image.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(image.tobytes(), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

                pixmap1 = QtGui.QPixmap.fromImage(qImg)

                size = QtCore.QSize(250, 560)
                self.image_label1.set_pixmap(pixmap1, size)
                self.image_label1.show()

        # Check if image2_path exists
        if image2_path is not None and os.path.exists(image2_path):
            size = QtCore.QSize(400, 400)
            pixmap2 = QtGui.QPixmap(image2_path)
            self.image_label2.set_pixmap(pixmap2, size)
            self.image_label2.show()
        else:
            self.image_label2.clear()
            self.image_label2.setText("Plant processing needed.")
            self.image_label2.setAlignment(QtCore.Qt.AlignCenter)
            self.image_label2.show()

        return


    def remove_selected_plant(self):
        path = self.selected_plant

        removed_path = self.projectField.text() + "/Removed"
        removed_path = os.path.join(removed_path, os.path.relpath(path, self.projectField.text() + "/Analysis"))

        if not os.path.exists(os.path.dirname(removed_path)):
            os.makedirs(os.path.dirname(removed_path))

        # Open the directory in the file explorer
        if os.name == 'nt':
            os.system(f'move "{path}" "{removed_path}"')
        elif sys.platform == 'darwin':
            os.system(f'mv "{path}" "{removed_path}"')
        else:
            os.system(f'mv "{path}" "{removed_path}"')
        
        self.refresh_table()
        
        return

    def analysis(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "1_analysis.py"])

    def getBBOX(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "1_analysis.py", "--getbbox"])

    def rerunAnalysis(self):
        metadata_path = os.path.join(self.selected_plant, "metadata.json")
        subprocess.Popen(["python", "1_analysis.py", "--config", metadata_path, "--rerun"])

    def rerunAnalysis_table(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = item.path

        metadata_path = os.path.join(path, "metadata.json")
        subprocess.Popen(["python", "1_analysis.py", "--config", metadata_path, "--rerun"])

    def preview(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "1_analysis.py", "--preview"])

    def PostProcess(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "2_postprocess.py"])
    
    def report(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "3_generateReport.py"])   

    def reviewPlant(self):
        path = self.selected_plant
        subprocess.Popen(["python", "4_reviewPlant.py", "--path", path])

    def drawAngles(self):
        path = self.selected_plant
        print("Not yet implemented")

    def syncProjectFolderField(self):
        projectFolder = self.projectField.text()
        projectFolder2 = self.projectField_2.text()

        if self.central_widget.sender() == self.projectField:
            self.projectField_2.setText(projectFolder)
        elif self.central_widget.sender() == self.projectField_2:
            self.projectField.setText(projectFolder2)
    
    def syncCaptureIntervalField(self):
        captureInterval = self.captureIntervalField.text()
        captureInterval2 = self.captureIntervalField_3.text()

        if self.central_widget.sender() == self.captureIntervalField:
            self.captureIntervalField_3.setText(captureInterval)
        elif self.central_widget.sender() == self.captureIntervalField_3:
            self.captureIntervalField.setText(captureInterval2)

    def syncProcessingLimitField(self):
        processingLimit = self.processingLimitField.text()
        processingLimit2 = self.processingLimitField_3.text()

        if self.central_widget.sender() == self.processingLimitField:
            self.processingLimitField_3.setText(processingLimit)
        elif self.central_widget.sender() == self.processingLimitField_3:
            self.processingLimitField.setText(processingLimit2)
                    
    def setupUi(self, chrono_root_analysis):
        chrono_root_analysis.setObjectName("ChronoRootAnalysis")
        chrono_root_analysis.resize(811, 600)
        self.central_widget = QtWidgets.QWidget(chrono_root_analysis)
        self.central_widget.setObjectName("centralwidget")
        
        self.setup_tabs()
        self.setup_tab1_elements()
        self.setup_tab2_elements()
        self.setup_tab3_elements()
        self.setup_tab4_elements()

        chrono_root_analysis.setCentralWidget(self.central_widget)
        self.statusbar = QtWidgets.QStatusBar(chrono_root_analysis)
        self.statusbar.setObjectName("statusbar")
        chrono_root_analysis.setStatusBar(self.statusbar)

        self.retranslate_ui(chrono_root_analysis)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(chrono_root_analysis)

        self.refresh_table()

    def setup_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget(self.central_widget)
        self.tab_widget.setGeometry(QtCore.QRect(0, 0, 811, 621))
        
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tab_widget.setFont(font)
        self.tab_widget.setObjectName("tabWidget")
        
        return
    
    def setup_tab1_elements(self):

        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.tab_widget.addTab(self.tab1, "")
    
        self.videoField = QtWidgets.QLineEdit(self.tab1)
        self.videoField.setGeometry(QtCore.QRect(190, 100, 441, 31))
        self.videoField.setObjectName("videoField")

        self.loadVideo = QtWidgets.QPushButton(self.tab1)
        self.loadVideo.setGeometry(QtCore.QRect(10, 100, 161, 31))
        self.loadVideo.setObjectName("loadVideo")
        self.loadVideo.clicked.connect(lambda: self.videoField.setText(self.openFileNameDialog()))

        self.loadProject = QtWidgets.QPushButton(self.tab1)
        self.loadProject.setGeometry(QtCore.QRect(10, 50, 161, 31))
        self.loadProject.setObjectName("loadProject")
        self.loadProject.clicked.connect(lambda: self.projectField.setText(self.openFileNameDialog()))

        self.projectField = QtWidgets.QLineEdit(self.tab1)
        self.projectField.setGeometry(QtCore.QRect(190, 50, 441, 31))
        self.projectField.setObjectName("projectField")
        self.projectField.textChanged.connect(self.syncProjectFolderField)

        self.rpiField = QtWidgets.QLineEdit(self.tab1)
        self.rpiField.setGeometry(QtCore.QRect(190, 150, 51, 31))
        self.rpiField.setObjectName("rpiField")

        self.cameraField = QtWidgets.QLineEdit(self.tab1)
        self.cameraField.setGeometry(QtCore.QRect(190, 200, 51, 31))
        self.cameraField.setObjectName("cameraField")

        self.plantField = QtWidgets.QLineEdit(self.tab1)
        self.plantField.setGeometry(QtCore.QRect(190, 250, 51, 31))
        self.plantField.setObjectName("plantField")

        self.identifierField = QtWidgets.QLineEdit(self.tab1)
        self.identifierField.setGeometry(QtCore.QRect(190, 300, 151, 31))
        self.identifierField.setObjectName("identifierField")

        self.saveImagesButton = QtWidgets.QCheckBox(self.tab1)
        self.saveImagesButton.setGeometry(QtCore.QRect(10, 400, 161, 31))
        self.saveImagesButton.setObjectName("saveImagesButton")

        self.captureIntervalField = QtWidgets.QLineEdit(self.tab1)
        self.captureIntervalField.setGeometry(QtCore.QRect(190, 500, 51, 31))
        self.captureIntervalField.setObjectName("captureIntervalField")
        self.captureIntervalField.textChanged.connect(self.syncCaptureIntervalField)

        self.processingLimitField = QtWidgets.QLineEdit(self.tab1)
        self.processingLimitField.setGeometry(QtCore.QRect(190, 450, 51, 31))
        self.processingLimitField.setObjectName("processingLimitField")
        self.processingLimitField.textChanged.connect(self.syncProcessingLimitField)

        self.emergenceDistanceField = QtWidgets.QLineEdit(self.tab1)
        self.emergenceDistanceField.setGeometry(QtCore.QRect(190, 550, 51, 31))
        self.emergenceDistanceField.setObjectName("emergenceDistanceField")

        self.saveButton = QtWidgets.QPushButton(self.tab1)
        self.saveButton.setGeometry(QtCore.QRect(660, 0, 141, 81))
        self.saveButton.setObjectName("saveButton")
        self.saveButton.clicked.connect(self.saveFieldsIntoJson)

        self.previewAnalysisButton = QtWidgets.QPushButton(self.tab1)
        self.previewAnalysisButton.setGeometry(QtCore.QRect(660, 100, 141, 81))
        self.previewAnalysisButton.setObjectName("previewAnalysisButton")
        self.previewAnalysisButton.clicked.connect(self.preview)

        self.analysisButton = QtWidgets.QPushButton(self.tab1)
        self.analysisButton.setGeometry(QtCore.QRect(660, 200, 141, 81))
        self.analysisButton.setObjectName("analysisButton")
        self.analysisButton.clicked.connect(self.analysis)

        self.PostProcessButton = QtWidgets.QPushButton(self.tab1)
        self.PostProcessButton.setGeometry(QtCore.QRect(660, 300, 141, 81))
        self.PostProcessButton.setObjectName("PostProcessButton")
        self.PostProcessButton.clicked.connect(self.PostProcess)

        self.loadLastConfigButton = QtWidgets.QPushButton(self.tab1)
        self.loadLastConfigButton.setGeometry(QtCore.QRect(660, 500, 141, 81))
        self.loadLastConfigButton.setObjectName("loadLastConfigButton")
        self.loadLastConfigButton.clicked.connect(self.loadJsonIntoFields)

        self.label = QtWidgets.QLabel(self.tab1)
        self.label.setGeometry(QtCore.QRect(10, 150, 161, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab1)
        self.label_2.setGeometry(QtCore.QRect(10, 200, 161, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab1)
        self.label_3.setGeometry(QtCore.QRect(10, 250, 161, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab1)
        self.label_4.setGeometry(QtCore.QRect(10, 300, 161, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab1)
        self.label_5.setGeometry(QtCore.QRect(260, 150, 261, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab1)
        self.label_6.setGeometry(QtCore.QRect(260, 200, 261, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab1)
        self.label_7.setGeometry(QtCore.QRect(260, 250, 261, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab1)
        self.label_8.setGeometry(QtCore.QRect(360, 300, 261, 31))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab1)
        self.label_9.setGeometry(QtCore.QRect(10, 360, 541, 31))
        self.label_9.setObjectName("label_9")
        self.label_11 = QtWidgets.QLabel(self.tab1)
        self.label_11.setGeometry(QtCore.QRect(10, 500, 161, 31))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab1)
        self.label_12.setGeometry(QtCore.QRect(10, 450, 161, 31))
        self.label_12.setObjectName("label_12")
        self.label_26 = QtWidgets.QLabel(self.tab1)
        self.label_26.setGeometry(QtCore.QRect(260, 450, 261, 31))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.tab1)
        self.label_27.setGeometry(QtCore.QRect(260, 500, 261, 31))
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.tab1)
        self.label_28.setGeometry(QtCore.QRect(190, 400, 441, 31))
        self.label_28.setObjectName("label_28")
        self.label_30 = QtWidgets.QLabel(self.tab1)
        self.label_30.setGeometry(QtCore.QRect(10, 10, 541, 31))
        self.label_30.setObjectName("label_30")
        self.line = QtWidgets.QFrame(self.tab1)
        self.line.setGeometry(QtCore.QRect(0, 340, 651, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.tab1)
        self.line_2.setGeometry(QtCore.QRect(640, -30, 20, 641))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.label_32 = QtWidgets.QLabel(self.tab1)
        self.label_32.setGeometry(QtCore.QRect(10, 550, 161, 31))
        self.label_32.setObjectName("label_32")

        self.label_33 = QtWidgets.QLabel(self.tab1)
        self.label_33.setGeometry(QtCore.QRect(260, 550, 261, 31))
        self.label_33.setObjectName("label_33")

        return

    def setup_tab2_elements(self):
        # Create the table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Variety", "Raspberry", "Camera", "Plant Number", "Result ID", 
                                              "Error Rate", "Status", "Finish Date"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Enable sorting
        self.table.setSortingEnabled(True)

        # Create the refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_table)

        # Create the rerun analysis button
        self.rerun_analysis_button_tab2 = QPushButton("Rerun Analysis")
        self.rerun_analysis_button_tab2.clicked.connect(self.rerunAnalysis_table)

        # Create the open path button
        self.open_path_button = QPushButton("Open Path")
        self.open_path_button.clicked.connect(self.open_selected_path)

        # Create the remove path button
        self.remove_path_button = QPushButton("Remove Plant")
        self.remove_path_button.clicked.connect(self.remove_selected_path)

        # Set up the layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.open_path_button)
        buttons_layout.addWidget(self.remove_path_button)
        buttons_layout.addWidget(self.rerun_analysis_button_tab2)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addLayout(buttons_layout)

        # Create and set up the new tab
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setLayout(layout)
        self.tab_widget.addTab(self.tab2, "Tab 2")


    def setup_tab3_elements(self):
        # Create the image labels
        self.image_label1 = AspectRatioLabel()
        self.image_label2 = AspectRatioLabel()

        # Set image labels to scale contents with aspect ratio
        self.image_label1.setMaximumSize(250, 560)
        self.image_label2.setMaximumSize(400, 400)

        # Create the checkbox
        self.overlay_checkbox = QCheckBox("Overlay Image")

        # Create the dropdown menu for plant selection to the right of the checkbox
        self.plant_dropdown = QComboBox()

        # Create the refresh button
        self.refresh_button_tab3 = QPushButton("Refresh_2")
        self.refresh_button_tab3.clicked.connect(self.refresh_table)

        # Create a rerun analysis button
        self.rerun_analysis_button = QPushButton("Rerun Analysis")
        self.rerun_analysis_button.clicked.connect(self.rerunAnalysis)

        # Create the remove path button
        self.remove_path_button_tab3 = QPushButton("Remove Plant")
        self.remove_path_button_tab3.clicked.connect(self.remove_selected_plant)

        # Connect signals
        self.plant_dropdown.currentIndexChanged.connect(self.update_image_labels)
        self.overlay_checkbox.stateChanged.connect(self.update_image_labels)

        self.reviewButton = QPushButton("View full sequence")
        self.reviewButton.clicked.connect(self.reviewPlant)

        self.drawAnglesButton = QPushButton("Draw angles")
        self.drawAnglesButton.clicked.connect(self.drawAngles)

        # Set up the layout for the checkbox, dropdown menu, and refresh button
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.overlay_checkbox)
        controls_layout.addWidget(self.plant_dropdown)

        controls_layout2 = QHBoxLayout()
        controls_layout2.addWidget(self.refresh_button_tab3)
        controls_layout2.addWidget(self.rerun_analysis_button)
        controls_layout2.addWidget(self.remove_path_button_tab3)
        controls_layout2.addWidget(self.reviewButton)
        controls_layout2.addWidget(self.drawAnglesButton)

        # Set up the main layout
        layout = QHBoxLayout()
        layout.addWidget(self.image_label1)
        layout.addWidget(self.image_label2)

        bigLayout = QVBoxLayout()
        bigLayout.addLayout(layout)
        bigLayout.addLayout(controls_layout)
        bigLayout.addLayout(controls_layout2)
        
        # Create and set up the new tab
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setLayout(bigLayout)
        self.tab_widget.addTab(self.tab3, "Plant Overlay")

    def setup_tab4_elements(self):
        self.tab4 = QtWidgets.QWidget()
        self.tab4.setObjectName("tab4")
        self.tab_widget.addTab(self.tab4, "")

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
    
        self.daysConvexField = QtWidgets.QLineEdit(self.tab4)
        self.daysConvexField.setGeometry(QtCore.QRect(150, 120, 221, 31))
        self.daysConvexField.setObjectName("daysConvexField")

        self.saveImagesConvex = QtWidgets.QCheckBox(self.tab4)
        self.saveImagesConvex.setGeometry(QtCore.QRect(10, 170, 611, 31))
        self.saveImagesConvex.setObjectName("saveImagesConvex")

        self.doConvex = QtWidgets.QCheckBox(self.tab4)
        self.doConvex.setGeometry(QtCore.QRect(10, 70, 301, 31))
        self.doConvex.setFont(font)
        self.doConvex.setObjectName("doConvex")

        self.doFourier = QtWidgets.QCheckBox(self.tab4)
        self.doFourier.setGeometry(QtCore.QRect(10, 240, 301, 31))
        self.doFourier.setFont(font)
        self.doFourier.setObjectName("doFourier")

        self.doLateralAngles = QtWidgets.QCheckBox(self.tab4)
        self.doLateralAngles.setGeometry(QtCore.QRect(10, 310, 301, 31))
        self.doLateralAngles.setFont(font)
        self.doLateralAngles.setObjectName("doLateralAngles")

        self.daysAnglesField = QtWidgets.QLineEdit(self.tab4)
        self.daysAnglesField.setGeometry(QtCore.QRect(150, 360, 221, 31))
        self.daysAnglesField.setObjectName("daysAnglesField")

        self.PostProcessButton2 = QtWidgets.QPushButton(self.tab4)
        self.PostProcessButton2.setGeometry(QtCore.QRect(360, 480, 131, 81))
        self.PostProcessButton2.setObjectName("PostProcessButton2")
        self.PostProcessButton2.clicked.connect(self.PostProcess)

        self.reportButton = QtWidgets.QPushButton(self.tab4)
        self.reportButton.setGeometry(QtCore.QRect(510, 480, 131, 81))
        self.reportButton.setObjectName("reportButton")
        self.reportButton.clicked.connect(self.report)

        self.loadLastConfig2 = QtWidgets.QPushButton(self.tab4)
        self.loadLastConfig2.setGeometry(QtCore.QRect(660, 480, 141, 81))
        self.loadLastConfig2.setObjectName("loadLastConfig2")
        self.loadLastConfig2.clicked.connect(self.loadJsonIntoFields)

        self.saveButton_2 = QtWidgets.QPushButton(self.tab4)
        self.saveButton_2.setGeometry(QtCore.QRect(210, 480, 131, 81))
        self.saveButton_2.setObjectName("saveButton_2")
        self.saveButton_2.clicked.connect(self.saveFieldsIntoJson)

        self.loadProject_2 = QtWidgets.QPushButton(self.tab4)
        self.loadProject_2.setGeometry(QtCore.QRect(10, 10, 161, 31))
        self.loadProject_2.clicked.connect(lambda: self.projectField.setText(self.openFileNameDialog()))

        self.projectField_2 = QtWidgets.QLineEdit(self.tab4)
        self.projectField_2.setGeometry(QtCore.QRect(190, 10, 441, 31))
        self.projectField_2.setObjectName("projectField_2")
        self.projectField_2.textChanged.connect(self.syncProjectFolderField)

        self.captureIntervalField_3 = QtWidgets.QLineEdit(self.tab4)
        self.captureIntervalField_3.setGeometry(QtCore.QRect(140, 530, 51, 31))
        self.captureIntervalField_3.setObjectName("captureIntervalField_3")
        self.captureIntervalField_3.textChanged.connect(self.syncCaptureIntervalField)

        self.processingLimitField_3 = QtWidgets.QLineEdit(self.tab4)
        self.processingLimitField_3.setGeometry(QtCore.QRect(140, 480, 51, 31))
        self.processingLimitField_3.setObjectName("processingLimitField_3")
        self.processingLimitField_3.textChanged.connect(self.syncProcessingLimitField)

        self.label_10 = QtWidgets.QLabel(self.tab4)
        self.label_10.setGeometry(QtCore.QRect(10, 120, 131, 31))
        self.label_10.setObjectName("label_10")
        self.label_29 = QtWidgets.QLabel(self.tab4)
        self.label_29.setGeometry(QtCore.QRect(380, 120, 351, 31))
        self.label_29.setObjectName("label_29")
        self.line_5 = QtWidgets.QFrame(self.tab4)
        self.line_5.setGeometry(QtCore.QRect(-40, 200, 891, 41))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self.tab4)
        self.line_6.setGeometry(QtCore.QRect(-70, 270, 961, 41))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.label_25 = QtWidgets.QLabel(self.tab4)
        self.label_25.setGeometry(QtCore.QRect(10, 360, 131, 31))
        self.label_25.setObjectName("label_25")
        self.label_31 = QtWidgets.QLabel(self.tab4)
        self.label_31.setGeometry(QtCore.QRect(380, 360, 351, 31))
        self.label_31.setObjectName("label_31")
        self.line_7 = QtWidgets.QFrame(self.tab4)
        self.line_7.setGeometry(QtCore.QRect(0, 40, 891, 41))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_44 = QtWidgets.QLabel(self.tab4)
        self.label_44.setGeometry(QtCore.QRect(10, 530, 111, 31))
        self.label_44.setObjectName("label_44")
        self.label_43 = QtWidgets.QLabel(self.tab4)
        self.label_43.setGeometry(QtCore.QRect(10, 480, 121, 31))
        self.label_43.setObjectName("label_43")
        self.line_11 = QtWidgets.QFrame(self.tab4)
        self.line_11.setGeometry(QtCore.QRect(-30, 440, 961, 41))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")

        return

    def retranslate_ui(self, ChronoRootAnalysis):
        _translate = QtCore.QCoreApplication.translate
        
        def set_translation(element, text):
            element.setText(_translate("ChronoRootAnalysis", text))

        def translate_main_elements():
            ChronoRootAnalysis.setWindowTitle(_translate("ChronoRootAnalysis", "ChronoRootAnalysis"))
            # Replace WIDTH and HEIGHT with the desired width and height of the window
            fixed_size = QtCore.QSize(810, 650)

            ChronoRootAnalysis.setMinimumSize(fixed_size)
            ChronoRootAnalysis.setMaximumSize(fixed_size)

            set_translation(self.loadVideo, "Select Video Folder")
            set_translation(self.loadProject, "Select Project Folder")
            set_translation(self.saveButton, "Save")

        def translate_labels():
            set_translation(self.label, "<html><head/><body><p align=\"center\">Raspberry Module</p></body></html>")
            set_translation(self.label_2, "<html><head/><body><p align=\"center\">Camera</p></body></html>")
            set_translation(self.label_3, "<html><head/><body><p align=\"center\">Plant Number</p></body></html>")
            set_translation(self.label_4, "<html><head/><body><p align=\"center\">Identifier</p></body></html>")
            set_translation(self.label_5, "(should be a number between 1-24)")
            set_translation(self.label_6, "(should be a number between 1-4)")
            set_translation(self.label_7, "(should be a number between 1-4)")
            set_translation(self.label_8, "(variety identifier, e.g. WT, Col0)")
            set_translation(self.label_9, "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Analysis and postprocessing parameters</span></p></body></html>")
            set_translation(self.label_11, "<html><head/><body><p>Capture interval</p></body></html>")
            set_translation(self.label_12, "<html><head/><body><p>Set processing limit</p></body></html>")
            set_translation(self.label_26, "(in days, 0 means no limit)")
            set_translation(self.label_27, "(in minutes, usually 15 minutes)")
            set_translation(self.label_28, "(useful to make growth videos, takes extra time and disk space)")
            set_translation(self.label_30, "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Individual plant root analysis</span></p></body></html>")
            set_translation(self.label_10, "Days to report")
            set_translation(self.label_29, "(Should be numbers separated by commas, e.g. 5,7,9,11)")
            set_translation(self.label_25, "Days to report")
            set_translation(self.label_31, "(Should be numbers separated by commas, e.g. 5,7,9,11)")
            set_translation(self.label_32, "Emergence distance")
            set_translation(self.label_33, "(in millimeters, recommended 1 or 2 mm)")
            set_translation(self.label_44, "<html><head/><body><p>Capture interval</p></body></html>")
            set_translation(self.label_43, "<html><head/><body><p>Processing limit</p></body></html>")

        def translate_buttons():
            set_translation(self.loadVideo, "Select Video Folder")
            set_translation(self.loadProject, "Select Project Folder")
            set_translation(self.saveButton, "Save")
            set_translation(self.loadLastConfigButton, "Load\nprevious\nconfiguration")
            set_translation(self.saveImagesButton, "Save Cropped Images")
            set_translation(self.analysisButton, "Analyze Plant")
            set_translation(self.previewAnalysisButton, "Preview video")
            set_translation(self.PostProcessButton, "Process\nall plants")
            set_translation(self.saveImagesConvex, "Save images for each day (if unselected will only save them for the last day)")
            set_translation(self.doConvex, "Do Convex hull analysis")
            set_translation(self.doFourier, "Do Fourier analysis")
            set_translation(self.doLateralAngles, "Do lateral root angles analysis")
            set_translation(self.PostProcessButton2, "Process\nall plants")
            set_translation(self.reportButton, "Generate report")
            set_translation(self.loadLastConfig2, "Load\nprevious\nconfiguration")
            set_translation(self.saveButton_2, "Save")
            set_translation(self.loadProject_2, "Select Project Folder")
            set_translation(self.refresh_button, "Refresh")
            set_translation(self.refresh_button_tab3, "Refresh")
            set_translation(self.open_path_button, "Open Path")

        def translate_tab_text():
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab1), _translate("ChronoRootAnalysis", "Plant Analysis"))
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab2), _translate("ChronoRootAnalysis", "Analysis Overview"))
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab4), _translate("ChronoRootAnalysis", "Generate Report"))
            
        translate_main_elements()
        translate_labels()
        translate_buttons()
        translate_tab_text()