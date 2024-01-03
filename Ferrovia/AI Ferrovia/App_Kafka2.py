import sys
import time
import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
import multiprocessing
import pyqtgraph as pg
from sklearn.ensemble import IsolationForest
from scipy.fft import fft, fftfreq
import scipy.stats as stats
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8
from datetime import datetime
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QGraphicsView, QMainWindow, QPushButton, QTableView, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui,QtCore
from PyQt5.QtCore import QModelIndex, QObject, QRunnable, QThread, QThreadPool, QTimer, QSemaphore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from threading import Thread, Lock

########## Global Variables MQTT ###########
MQTT_ClIENT = mqtt.Client()
MQTT_BROKER_ADDRESS = 'localhost'
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "main_file"
############################################

######## Global Variables Dataframe ########
DF_MAIN_DATAFRAME = None
LASTDATA = None
PREVIOUS = None
PREVIOUS_DF = None
MAIN_DIR = "C:\\Users\\steam\\Desktop\\Bolsa\\Ferrovia\\AI Ferrovia\\"
COLUMNS_NAMES = ["Acc_Axlebox_Back_Left","Acc_Axlebox_Back_Right","Acc_Axlebox_Front_Left","Acc_Axlebox_Front_Right","Acc_Carbody_Back","Acc_Carbody_Front","Acc_Carbody_Middle"]
DF_CURRENT = None
############################################
semaphore = Lock()
######## Global Variables DELAY #########
DELAY_CLASSIFICATION = 0
DELAY_TABLE = 0
DELAY_PLOT = 0
######## Global Variables Counters #########
#COUNTER_INITIAL_STEP = 1000
MULTIPLIER = 1000
CURRENT_TOPIC = None
COUNTER_STEP = 1000
CURRENT_NUMBER_MESSAGES = 0 # Número de menssagens de mqtt subscritas             
############################################

########### Global Variables AI ############
MODEL = None
Simulation = None
############################################

######## Global Variables Semaphore ########
#DONE = 0
CONNECT = 0 # 0 está disconectado no mqtt 1 está conectado 
CHANGE = 0
FIRST = 0
WORK = 0 # Enquanto esta variável está a 0 não se faz o cálculo de AI quando for 1 trabalha-se
############################################
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data
        self._sort_col = None
        self._sort_order = QtCore.Qt.AscendingOrder

    def rowCount(self, parent=None):
        return len(self._data.index)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return QtCore.QVariant()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])
            elif orientation == QtCore.Qt.Vertical:
                return str(self._data.index[section])
        return QtCore.QVariant()

    def sort(self, col, order):
        self.layoutAboutToBeChanged.emit()
        self._sort_col = col
        self._sort_order = order
        if order == QtCore.Qt.AscendingOrder:
            self._data = self._data.sort_values(self._data.columns[col], ascending=True)
        else:
            self._data = self._data.sort_values(self._data.columns[col], ascending=False)
        self.layoutChanged.emit()

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role):
        if index.isValid() and role == QtCore.Qt.EditRole:
            row = index.row()
            col = index.column()
            self._data.iat[row, col] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def appendRows(self, new_data):
        self.beginInsertRows(QtCore.QModelIndex(), len(self._data.index), len(self._data.index) + len(new_data.index) - 1)
        self._data = pd.concat([self._data, new_data], axis=0, ignore_index=True)
        self._data.sort_values(by="Index", ascending=False, inplace=True)
        self._data.reset_index(drop=True, inplace=True) # reindex the DataFrame
        self.endInsertRows()
        self.layoutChanged.emit()
 
def features_extraction(df, FEATURES): 
        Min=[];Max=[];Mean=[];Rms=[];Var=[];Std=[];Power=[];Peak=[];Skew=[];Kurtosis=[];P2p=[];CrestFactor=[]
        FormFactor=[]; PulseIndicator=[]
        Max_f=[];Sum_f=[];Mean_f=[];Var_f=[];Peak_f=[];Skew_f=[];Kurtosis_f=[];length=[];Margin=[];Margin_f=[];P2p_f=[];Min_f=[]
        
        '''
        ;RMS_LOW=[];RMS_MED=[];RMS_HIGH=[]
        '''
        
        X = np.diff(df.values)
        ## TIME DOMAIN ##
        length.append(len(X))
        Min.append(np.min(X))
        Max.append(np.max(X))
        Mean.append(np.mean(X))
        Rms.append(np.sqrt(np.mean(X**2)))
        Var.append(np.var(X))
        Std.append(np.std(X))
        Power.append(np.mean(X**2))
        Peak.append(np.max(np.abs(X)))
        P2p.append(np.ptp(X))
        CrestFactor.append(np.max(np.abs(X))/np.sqrt(np.mean(X**2)))
        Skew.append(stats.skew(X))
        Kurtosis.append(stats.kurtosis(X))
        FormFactor.append(np.sqrt(np.mean(X**2))/np.mean(X))
        PulseIndicator.append(np.max(np.abs(X))/np.mean(X))
        Margin.append(np.max(np.abs(X))/(np.abs(np.mean(np.sqrt(np.abs(X))))**2))
        ## FREQ DOMAIN ##
        ft = fft(X)
        S = np.abs(ft**2)/len(df)
        Max_f.append(np.max(S))
        Min_f.append(np.min(S))
        Sum_f.append(np.sum(S))
        Mean_f.append(np.mean(S))
        Var_f.append(np.var(S))
        P2p_f.append(np.ptp(S))
        Peak_f.append(np.max(np.abs(S)))
        Skew_f.append(stats.skew(S))
        Kurtosis_f.append(stats.kurtosis(S))
        Margin_f.append(np.max(np.abs(S))/(np.abs(np.mean(np.sqrt(np.abs(S))))**2))
        '''
        RMS_LOW.append(np.sqrt(np.mean(S[0:int(len(S) / 4)]**2))) 
        RMS_MED.append(np.sqrt(np.mean(S[int(len(S) / 4):int(3 * len(S) / 4)]**2))) 
        RMS_HIGH.append(np.sqrt(np.mean(S[int(3 * len(S) / 4):]**2))) 
        '''
        #Create dataframe from features
        df_features = pd.DataFrame(index = [FEATURES], 
                                data = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis,
                                        Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f,length,Margin,Margin_f,P2p_f,Min_f,FormFactor,PulseIndicator])
        return df_features

  
def classification(semaphore):
    
    global DF_MAIN_DATAFRAME
    global COUNTER_STEP
    global MULTIPLIER
    global Simulation
    global PREVIOUS

    
    aux = DF_MAIN_DATAFRAME
    DF_MAIN_DATAFRAME = DF_MAIN_DATAFRAME.tail(500)
    
    

    try:
        #print(3)
        #print(len(aux))
        #len(aux)

        #if len(aux) >= MULTIPLIER:
        #print("Length ->" + str(COUNTER_STEP))
        #thread_id = int(QThread.currentThreadId())
        #print("Current thread ID:", thread_id)
        
        start = MULTIPLIER-1000
        end = MULTIPLIER
        MULTIPLIER = 1000
        #print(start)
        #print(end)
        #print(COUNTER_STEP)
        window = aux.iloc[start:end,:]
        COUNTER_STEP += 500
        print(str(len(window)))
        
        #print("Start -> " + str(start))
        #print("End -> " + str(end))
        #print("Counter -> " + str(COUNTER_STEP))
        data = {'Cenario':  [aux.Cenario[start]]}
        features = pd.DataFrame(data).T 
        
        for i in range(1,8):
            features_new = ['MIN_{j}'.format(j=i),'MAX_{j}'.format(j=i),'MEAN_{j}'.format(j=i),'RMS_{j}'.format(j=i),'VAR_{j}'.format(j=i),'STD_{j}'.format(j=i),'POWER_{j}'.format(j=i),'PEAK_{j}'.format(j=i),'P2P_{j}'.format(j=i),'CREST_FACTOR_{j}'.format(j=i),'SKEW_{j}'.format(j=i),'KURTOSIS_{j}'.format(j=i),
                    'MAX_f_{j}'.format(j=i),'SUM_f_{j}'.format(j=i),'MEAN_f_{j}'.format(j=i),'VAR_f_{j}'.format(j=i),'PEAK_f_{j}'.format(j=i),'SKEW_f_{j}'.format(j=i),'KURTOSIS_f_{j}'.format(j=i), 'len_{j}'.format(j=i), 'Margin_{j}'.format(j=i), 'Margin_f_{j}'.format(j=i), 'P2p_f_{j}'.format(j=i), 'Min_f_{j}'.format(j=i), 'FORM_FACTOR_{j}'.format(j=i), 'PULSE_INDICATOR_{j}'.format(j=i)]
            features = pd.concat([features,features_extraction(window.iloc[:,int(i-1)], features_new)])
        #print("Tamanho features ->" + str(len(features)))
        add = features.T

        add = add.rename(columns={col: col[0] for col in add.columns if isinstance(col, tuple)})
        add['Score'] = Model.decision_function(add.iloc[:,1:183])
        #print(add['Score'][0])
        a = add.PEAK_1[0]
        a1 = add.MAX_1[0]
        b = add.PEAK_2[0]
        b1 = add.MAX_2[0]
        c = add.PEAK_3[0]
        c1 = add.MAX_3[0]
        d = add.PEAK_4[0]
        d1 = add.MAX_4[0]
        if a > b and a > c and a > d:
            add['Lado'] = 'Esquerda'
            if a == a1:
                add['Tipo'] = 'Depressão'
            else:
                add['Tipo'] = 'Lomba'    
        elif b > a and b > c and b > d:
            add['Lado'] = 'Direita'
            if b == b1:
                add['Tipo'] = 'Depressão'
            else:
                add['Tipo'] = 'Lomba'
        elif c > a and c > b and c > d:
            add['Lado'] = 'Esquerda'
            if c == c1:
                add['Tipo'] = 'Depressão'
            else:
                add['Tipo'] = 'Lomba'
        else:
            add['Lado'] = 'Direita'
            if d == d1:
                add['Tipo'] = 'Depressão'
            else:
                add['Tipo'] = 'Lomba'
                    
        add['Index'] = COUNTER_STEP-500
        #print(COUNTER_STEP)
        if COUNTER_STEP == 1000:
            Simulation = add.copy()
            #print("Tamanho Simulação -> " + str(len(Simulation)))
        else:
            Simulation = pd.concat([Simulation,add]).sort_values(by=['Index'], ascending=False)
            #Simulation = Simulation
        print(str(COUNTER_STEP))
        #else:
            #semaphore.release()
        #    return None
        
    except Exception as e:
        print(f'Classification Function -> {e}')
        return None
    #print(4)
    
def reset_all():
        global DF_MAIN_DATAFRAME
        global DF_CURRENT
        global CURRENT_NUMBER_MESSAGES
        global FIRST
        global CHANGE
        global CURRENT_TOPIC
        global COUNTER_STEP
        global MULTIPLIER
        global Simulation
        FIRST=0
        Simulation=None
        CURRENT_NUMBER_MESSAGES=0
        DF_MAIN_DATAFRAME=None
        CHANGE=0
        DF_CURRENT=None
        CURRENT_TOPIC = "Test"
        MULTIPLIER=1000
        COUNTER_STEP=1000

class ClassificationThread(QThread):
    def __init__(self,semaphore ,parent=None):
        super().__init__(parent)
        self.semaphore = semaphore
    def run(self):
        try:
            semaphore.acquire()
            classification(self.semaphore)
            semaphore.release()
        except:
            pass
        
            
            
class WorkerThread(QThread):
    def __init__(self,main_class, parent=None):
        super().__init__()
        self.main_class = main_class
        self.timer = QTimer()
        self.timer.setInterval(DELAY_PLOT)
        self.timer.timeout.connect(self.main_class.update_plot)
        self.thread = QThread()
        self.timer.moveToThread(self.thread)
        

    def stop(self):
        self.timer.stop()
        self.thread.quit()
        self.thread.wait()

    def run(self):
        self.thread.started.connect(self.timer.start)
        self.thread.start()
    
class UpdateTableWorker(QThread):
    def __init__(self, table_model, parent=None):
        super().__init__()
        self.table_model = table_model
    
    def run(self):
        try:
            self.msleep(200)
            global Simulation
            global LASTDATA
            df = Simulation
            #print(str(len(Simulation)))
            df_new = df[["Index","Lado","Tipo","Score"]]
            # Fetch the updated data
            if LASTDATA is not None and LASTDATA.equals(df_new):
                return  # Don't update if data is unchanged
            df_new = df_new.drop_duplicates(subset=['Index'])
            LASTDATA = df_new
            #print("Tamanho SIMULATON ->" + str(len(Simulation)))
            model = PandasModel(df_new)
            self.table_model.setModel(model)
            return
        except Exception as e:
            print(e)
            return
            
        

class MainWindow(QMainWindow):
    def __init__(self):
        global MAIN_DIR
        global DELAY_PLOT
        global DELAY_TABLE
        super().__init__()

        self.setWindowTitle("Ferrovia 4.0 Dashboard Edge")
        self.setWindowIcon(QIcon(MAIN_DIR+"icon.png"))
        self.setGeometry(100, 100, 1100, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.last_data = None
        layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.btn = QPushButton('Save Data', self)
        

        layout.addLayout(left_layout)

        table_image_layout = QHBoxLayout()

        self.image_label = QLabel()
        pixmap = QPixmap(MAIN_DIR + "train.png")
        self.image_label.setPixmap(pixmap)

        self.dropdown = QComboBox()
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)

        df = pd.DataFrame(columns=["Index", "Lado", "Tipo", "Score"])
        self.model = PandasModel(df)
        # Reset the model in the QTableView widget
        self.table_view.setModel(self.model)

        left_layout.addWidget(self.table_view)
        left_layout.addLayout(table_image_layout)
        table_image_layout.addWidget(self.image_label)

        layout.addLayout(right_layout)

        self.plotWidget = pg.PlotWidget(title='Real-Time Plot')
        self.plotWidget.setLabel('bottom', 'X Time 1 -> 0,0002 S')
        self.plotWidget.setLabel('left', 'Acceleration')
        self.plotWidget.showGrid(False, False)

        self.plotWidget.setLimits(xMin=0, yMin=-5, yMax=5, minXRange=0, maxXRange=5000)
        self.plotWidget.setYRange(-5, 5)
        self.curve = self.plotWidget.plot(pen=pg.mkPen('w', width=1))
        right_layout.addWidget(self.plotWidget)

        right_layout.addWidget(self.dropdown)
        right_layout.addWidget(self.btn)

        self.dropdown.addItems(COLUMNS_NAMES)
        self.dropdown.currentIndexChanged.connect(self.dropdown_changed)

        self.btn.clicked.connect(self.on_click)
        
        #self.timer = QTimer()
        #self.timer.setInterval(DELAY_PLOT)
        #self.timer.timeout.connect(self.update_plot)
        #self.timer.start()
        #self.classification_thread1 = WorkerThread(self)
        #self.classification_thread1.start()
        self.timer = QTimer()
        self.timer.setInterval(DELAY_PLOT)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        #self.thread = QThread()
        #self.timer.moveToThread(self.thread)
        #self.thread.started.connect(self.timer.start)
        #self.thread.start()
        #self.classification_timer = QTimer()
        #self.classification_timer.setInterval(1000)  # 1 second
        #self.classification_timer.timeout.connect(classification)
        #self.classification_timer.start()
        self.semaphore = QSemaphore(1)
        #self.classification_thread = ClassificationThread(self.semaphore)
        #self.classification_thread2 = ClassificationThread(self.semaphore)
        #self.classification_thread3 = ClassificationThread(self.semaphore)
        
        #self.classification_thread.start()
        #self.classification_thread2.start()
        #self.classification_thread3.start()

        self.table_timer = QTimer()
        self.table_timer.setInterval(DELAY_TABLE)  # 1 second
        self.table_timer.timeout.connect(self.update_table)
        self.table_timer.start()
        reset_all()
        

        
        
    def on_click(self):
        global DF_MAIN_DATAFRAME
        global MAIN_DIR
        global Simulation
        now = datetime.now()
        formatted_date = now.strftime("%d-%m-%Y-%H-%S")
        DF_MAIN_DATAFRAME.to_csv(MAIN_DIR+formatted_date+'_Real_Time.csv')
        Simulation.to_csv(MAIN_DIR+formatted_date+'_Problemas.csv')

    def dropdown_changed(self, index):
        print("Índice selecionado: ", index)

    def update_table(self):
    #    self.thread_pool.start(self.update_worker)
        global Simulation
        df = Simulation
        #print(str(len(Simulation)))
        df_new = df[["Index","Lado","Tipo","Score"]]
        # Fetch the updated data
        if Simulation is None:
            return
        #if self.last_data is not None and self.last_data.equals(df_new):
            #return  # Don't update if data is unchanged
        df_new = df_new.drop_duplicates(subset=['Index'])
        #self.last_data = df_new
        #
        
        #print("Tamanho SIMULATON ->" + str(len(Simulation)))
        #if PRIMEIRO ==0:
        #    model = PandasModel(df_new)
        #    self.table_view.setModel(model)
        #else:
        self.model.appendRows(df_new)
        Simulation=None

        
    def update_plot(self):
        #global DONE
        global DF_CURRENT
        global FIRST
        global MULTIPLIER
        global COUNTER_STEP
        global CHANGE
        global CURRENT_TOPIC
        global DF_MAIN_DATAFRAME
        global CURRENT_NUMBER_MESSAGES
        global PREVIOUS
        global Simulation
        
        
        #DONE = 1
        window_size = 2000
        old_data_percentage = 0
        
        try:
            if DF_CURRENT is None:
                return
        except:
            pass
        while CHANGE == 0:
            pass
        
        #CHANGE = 0
        column = self.dropdown.currentText()
        
        if CURRENT_TOPIC is None:
            CURRENT_TOPIC = column
        elif CURRENT_TOPIC != column:
            print("CHANGED")
            reset_all()
            CURRENT_TOPIC = column

            #### Testar a ver se atualiza com none
            self.curve.setData([], [])
            df = pd.DataFrame(columns=["Index", "Lado", "Tipo", "Score"])
            self.model = PandasModel(df)
            # Reset the model in the QTableView widget
            self.table_view.setModel(self.model)

            #######################################
            return
        
        
        # Generate random data
        self.plotWidget.setTitle('Real-Time Plot ' + CURRENT_TOPIC.replace("Acc_", ""))
        x_old = self.curve.xData
        y_old = self.curve.yData
        #print(x_old)
        #print(y_old)
        
        try:
            if PREVIOUS.equals(DF_CURRENT[column]):
                return 
        except:
            pass
        
        PREVIOUS = DF_CURRENT[column]
        DF_MAIN_DATAFRAME = pd.concat([DF_CURRENT, DF_MAIN_DATAFRAME], ignore_index=True)
        if FIRST == 0:
            y = DF_CURRENT[column]
            x = np.arange(len(y))
            FIRST += 1
        else:
            y_new = DF_CURRENT[column]
            x_new = np.arange(x_old[-1] + 1, x_old[-1] + len(y_new) + 1)
            
            # Check if we need to start a new window
            
            if len(x_old) > 0 and x_new[0] - x_old[-1] > window_size:
                num_old = int(len(x_old) * old_data_percentage)
                x_old = x_old[-num_old:]
                y_old = y_old[-num_old:]
            
            x = np.concatenate((x_old, x_new))
            y = np.concatenate((y_old, y_new))
            
            # Keep only the most recent data points
            #if len(x) > window_size:
            #    num_old = int(len(x) * old_data_percentage)
            #    x = x[-num_old-window_size:]
            #    y = y[-num_old-window_size:]
        
        # Update the plot
        
        self.curve.setData(x, y)
        
        # Only autoscale the x-axis when we reach a certain point
        if x[-1] >= window_size:
            self.plotWidget.setXRange(x[-1] - window_size, x[-1])
        else:
            self.plotWidget.setXRange(0, window_size)
        
        #self.plotWidget.enableAutoRange('y', True)
        
        #QApplication.processEvents()
        DONE = 0
        #semaphore2.release()
        try:
           # print("Main ->"+str(MULTIPLIER)+"_____"+str(len(DF_MAIN_DATAFRAME)))
            if len(DF_MAIN_DATAFRAME) > MULTIPLIER:
                
                #print("Executout no ->"+str(COUNTER_STEP))
                #print(CURRENT_NUMBER_MESSAGES*10)
                #CURRENT_NUMBER_MESSAGES = 0
                self.classification_thread = ClassificationThread(self.semaphore)
                self.classification_thread.start()
                #self.update_worker = UpdateTableWorker(self.table_view)
                #self.update_worker.start()
        except:
            #semaphore2.release()
            #semaphore3.release()
            #print("LEFT PLOT")
            pass
        
        #semaphore3.release()
        #print("LEFT PLOT")
        

    

        ##################################################
        

class MqttThread(multiprocessing.Process):
    def __init__(self):
        super().__init__()
        MQTT_ClIENT.on_message = self.on_message
        MQTT_ClIENT.on_connect = self.on_connect
        try:
            MQTT_ClIENT.connect(MQTT_BROKER_ADDRESS,MQTT_BROKER_PORT)
            MQTT_ClIENT.subscribe(MQTT_TOPIC)
            MQTT_ClIENT.loop_start()
        except Exception as e:
            print(f"[!] Error establishing connection -> {e}")

    def on_connect(self,client, userdata, flags, rc):
        global CONNECT
        CONNECT = 1
        print(f"[+] Connection Established -> {CONNECT}")

    def on_message(self,client, userdata, message):
        global DF_MAIN_DATAFRAME
        global CHANGE
        #global DONE
        global CURRENT_NUMBER_MESSAGES
        global DF_CURRENT
        
        ######### Tratar da menssagem em Dataframe #########
        payload = message.payload.decode("utf-8")
        ########## Já acabou o Loop do Main File? ##########
        #Refresh a tudo
        if payload == "finish" or CURRENT_NUMBER_MESSAGES >= 5000:
            print("finish")
            reset_all()
            
        else:
        ####################################################
            df = pd.read_json(payload)
            #print(df)
            #if DONE == 0:
            
            DF_CURRENT = df
            #else:
            #    DF_CURRENT = pd.concat([df, DF_CURRENT], ignore_index=True)
            #else:
                #DF_CURRENT = pd.concat([df, DF_CURRENT], ignore_index=True)
            #DF_MAIN_DATAFRAME = pd.concat([df, DF_MAIN_DATAFRAME], ignore_index=True)
            #print(str(len(DF_CURRENT)))
            CHANGE = 1
        
        #print("LEFT ON MESSAGE")
        time.sleep(0.001)

        #print(CHANGE)
        ####################################################
        #print("current messages ->"+str(CURRENT_NUMBER_MESSAGES))
        ######### Tratamento de contadores gerais ########## !!!! MOVING TO CLASSIFICATION SECTION !!!!!
        CURRENT_NUMBER_MESSAGES+=1 # Só porque sim
        #current_lines = len(DF_MAIN_DATAFRAME)
        #if current_lines % 1000 == 0:
        #    COUNTER_INITIAL_STEP = current_lines
        #    COUNTER_STEP = COUNTER_INITIAL_STEP-500
        #    print(COUNTER_INITIAL_STEP)
        #    print(COUNTER_STEP)
        #    WORK=1
        #print(len(DF_MAIN_DATAFRAME))
        
        #################################################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        
    

    def run(self):
        while True:
            # Código para atualizar o plot aqui
            pass

#@cuda.jit
def train_model():
        global Model
        dataset = pd.read_excel(MAIN_DIR+"Ferrovia.xlsx")
        dataset = dataset.drop('Unnamed: 0', axis = 1)
        df = dataset.iloc[:,2:184]

        model = IsolationForest()
        model.fit(df)
        # Make predictions
        y_pred = model.predict(df)
        scores = model.decision_function(df)
        #dataset['scores'] = scores
        
        return model



if __name__ == "__main__":
    global Model
    #p = multiprocessing.Process(target=train_model, args=('bob',))
    #p = multiprocessing.Process(target=train_model)
    #p.start()
    model = train_model()
    Model = model
    print(Model)
    #time.sleep(5)
    #print(Model)
    mqqt_thread = MqttThread()
    mqqt_thread.start()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    