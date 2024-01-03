import multiprocessing
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QComboBox, QTreeView, QTableView, QVBoxLayout, QHBoxLayout, QAbstractItemView, QHeaderView, QPushButton
from PyQt5.QtGui import QStandardItemModel, QStandardItem,QCursor, QPixmap,QFont,QColor,QBrush,QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import QModelIndex, QTimer, Qt,QAbstractTableModel,QAbstractItemModel,QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
from scipy.fft import fft, fftfreq
import scipy.stats as stats
import numpy as np
import pandas as pd
import sys
import os
import time
import json
import paho.mqtt.client as mqtt

MAIN_DIR = "C:\\Users\\steam\\Desktop\\Bolsa\\Ferrovia\\data"
REFRESH_RATE_SECS = 0.2
Model = None
REFRESH_RATE = 0
REFRESH_AI = 0
REFRESH_RATE_SECS_COMPONENT = 0
REFRESH_RATE_COMPONENT = REFRESH_RATE_SECS_COMPONENT * 1000

mqtt_broker_address = 'localhost'
mqtt_broker_port = 1883
DF_MAIN_PLOT = None
CURRENT_TOPIC = None
CHANGE = False
SUBSCRIBE=0
CONNECT=0



class CustomTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
        self._headers = data.columns.tolist()
        
        

    def rowCount(self, parent):
        return len(self._data.index)

    def columnCount(self, parent):
        return len(self._data.columns)

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._headers[section])
            elif orientation == QtCore.Qt.Vertical:
                return str(self._data.index[section])
        return None




def Get_SubComponents_Names():
    global DF_MAIN_PLOT
    df = DF_MAIN_PLOT
    
    column_names = df.columns.tolist()
    
    #print(column_names)
    new_arr = column_names
    #print([s[:-2] for s in new_arr])
    return [s for s in new_arr]
    

def get_df_column(parent_text,parent_parent,item_text):
    global DF_MAIN_PLOT
    #csv_name = MAIN_DIR+"\\"+parent_text+".csv"
    column_name = ""
    df = None
    column_name = item_text
    df = DF_MAIN_PLOT[[column_name]]
    return df

def get_df_full(cols):
    global DF_MAIN_PLOT
    #print(DF_MAIN_PLOT.columns)
    df = DF_MAIN_PLOT
    #print(df.columns)
    #csv_name = MAIN_DIR+"\\"+parent_text+".csv"
    df = df.loc[:, cols]
    return df




        
class SelectionThread(QThread):
    selection_changed = pyqtSignal()

    def __init__(self, selection_model):
        super().__init__()
        self.selection_model = selection_model

    def run(self):
        self.selection_model.selectionChanged.connect(self.emit_selection_changed)

    def emit_selection_changed(self):
        self.selection_changed.emit()

class MyWindow(QWidget):
    def __init__(self):
        global mqtt_broker_address
        global mqtt_broker_port

        self.model = QStandardItemModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers) # impedindo a edição dos itens
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents) # ajustando o tamanho das colunas
        self.tree.setExpandsOnDoubleClick(False) # impedindo que os itens se expandam ao serem clicados duas vezes
        
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.connect(mqtt_broker_address, mqtt_broker_port)  
        super().__init__()

        # Criando os componentes do combóio
        self.combobox = QComboBox()
        
        self.timer2 = QTimer()
        self.Get_Components_Names()
        # Add image 
        
        self.setStyleSheet("color: #333633;")
        pixmap = QPixmap(MAIN_DIR.replace("\\data","")+"\\"+"train.png")
        #pixmap2 = QPixmap(MAIN_DIR+"\\"+"legend.png")
        label = QLabel(self)
        #label2 = QLabel(self)
        label.setPixmap(pixmap)
        
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        
        self.ax.set_xlabel('Time 1 - 0,0002 secs')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Data Plot')
        
        # Criando a lista de subcomponentes
        
        
        self.tree.setHeaderHidden(True) # adicionando esta linha
        # Criando a tabela com os dados
        self.table = QTableView()
        
        # Adicionando os componentes ao layout
        button = QPushButton("Refresh")
        button.setFont(QFont("Calibri", 10, QFont.Bold))
        button.clicked.connect(self.Get_Components_Names)
        vbox = QVBoxLayout()
        vbox.addWidget(button)
        vbox.addWidget(self.combobox)
        vbox.addWidget(self.figure.canvas)
        vbox.addWidget(self.tree)
        vbox.addWidget(label)
        

        

        #self.tree.setStyleSheet("background-color:#E8ECEF;color: #333633;")
        self.tree.setFont(QFont("Calibri", 10, QFont.Bold))
        #self.combobox.setStyleSheet("background-color:#E8ECEF;color: #333633;")
        self.combobox.setFont(QFont("Calibri", 10, QFont.Bold))
        self.setFont(QFont("Calibri", 10, QFont.Bold))
        hbox = QHBoxLayout()
        hbox.addWidget(self.table)
        hbox.addLayout(vbox)
        
        self.setLayout(hbox)

        # Conectando o evento de seleção do combobox

        self.combobox.currentIndexChanged.connect(self.update_subcomponent_list)
        
        self.selection_model = self.tree.selectionModel()
        self.thread = SelectionThread(self.selection_model)
        #selection_model.selectionChanged.connect(self.show_selected_items)
        self.thread.selection_changed.connect(self.update_show_selected)
        self.thread.start()

        #self.timer = QtCore.QTimer()
        #self.timer.setInterval(REFRESH_RATE)  # interval in milliseconds

        # connect timer to update_data function
        #self.timer.timeout.connect(self.show_selected_items)

        # start the timer
        #self.timer.start()
        
    def features_extraction(self,df, FEATURES): 
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
    
    def train_model(self):
        global Model
        dataset = pd.read_excel("Ferrovia.xlsx")
        dataset = dataset.drop('Unnamed: 0', axis = 1)
        df = dataset.iloc[:,2:184]

        model = IsolationForest()
        model.fit(df)
        # Make predictions
        y_pred = model.predict(df)
        scores = model.decision_function(df)
        #dataset['scores'] = scores
        
        Model = model

    def classification(self):
        DF_MAIN_PLOT
        aux = DF_MAIN_PLOT

        
        window = aux.iloc[start:end,:]
        p = p + 1



        data = {'Cenario':  [aux.Cenario[start]]}
        features = pd.DataFrame(data).T 
        for i in range(1,8):
            features_new = ['MIN_{j}'.format(j=i),'MAX_{j}'.format(j=i),'MEAN_{j}'.format(j=i),'RMS_{j}'.format(j=i),'VAR_{j}'.format(j=i),'STD_{j}'.format(j=i),'POWER_{j}'.format(j=i),'PEAK_{j}'.format(j=i),'P2P_{j}'.format(j=i),'CREST_FACTOR_{j}'.format(j=i),'SKEW_{j}'.format(j=i),'KURTOSIS_{j}'.format(j=i),
                    'MAX_f_{j}'.format(j=i),'SUM_f_{j}'.format(j=i),'MEAN_f_{j}'.format(j=i),'VAR_f_{j}'.format(j=i),'PEAK_f_{j}'.format(j=i),'SKEW_f_{j}'.format(j=i),'KURTOSIS_f_{j}'.format(j=i), 'len_{j}'.format(j=i), 'Margin_{j}'.format(j=i), 'Margin_f_{j}'.format(j=i), 'P2p_f_{j}'.format(j=i), 'Min_f_{j}'.format(j=i), 'FORM_FACTOR_{j}'.format(j=i), 'PULSE_INDICATOR_{j}'.format(j=i)]
            features = features.append(self.features_extraction(window.iloc[:,int(i-1)], features_new))
        add = features.T

        add = add.rename(columns={col: col[0] for col in add.columns if isinstance(col, tuple)})
        add['Score'] = Model.decision_function(add.iloc[:,1:183])
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
                    
        if start == 0:
            Simulation = add.copy()
        else:
            Simulation = Simulation.append(add)

        start += step_size
        end += step_size
    def on_connect(self,client, userdata, flags, rc):
        global CONNECT
        CONNECT = 1
        print(CONNECT)

    def on_message(self,client, userdata, message):
        global DF_MAIN_PLOT
        global CHANGE

    # Convert the message payload from bytes to string
        message_str = message.payload.decode('utf-8')

        # Parse the JSON string into a list of rows with column names
        rows_with_header = json.loads(message_str)
        #print(rows_with_header)
        # Convert the list of rows with column names to a pandas dataframe
        df2 = pd.DataFrame(rows_with_header[1:], columns=rows_with_header[0])
        #cols = df.columns.tolist()
        # Print the pandas dataframe
        #print(df2.columns)

        if CHANGE != True:
            DF_MAIN_PLOT = pd.concat([DF_MAIN_PLOT,df2],ignore_index=True)
            #print(DF_MAIN_PLOT)
            #print(DF_MAIN_PLOT.columns)
        else:
            DF_MAIN_PLOT = df2
            #print(df.columns)
            CHANGE = False
            
    def Get_Components_Names(self):
        #cwd = os.getcwd()
        names = []
        with open(MAIN_DIR+"\\"+'topics.txt', 'r') as f:
        # Read the lines and store them in a list
            names = [line.strip() for line in f.readlines()]

        arr = []
        for i in range(self.combobox.count()):
            arr.append(self.combobox.itemText(i))
        for i in names:
            if i not in arr:
                self.combobox.addItem(i)
        for i in range(self.combobox.count()):
            if self.combobox.itemText(i) not in names:
                self.combobox.removeItem(i)
        self.model.clear()
        #self.combobox.addItems(names)
        
        
    def update_plot(self, x, y):
        # Atualiza os dados do gráfico e o redesenha
        WINDOW = 20000
        PERCENTAGE = 0.5
        xstart = max(0, len(x) - WINDOW)
        xend = len(x)
        xdata_window = x[xstart:xend]
        ydata_window = y[xstart:xend]

        # Atualiza os dados do gráfico e o redesenha
        old_data_idx = int(len(xdata_window) * PERCENTAGE)
        self.line.set_data(xdata_window[:old_data_idx], ydata_window[:old_data_idx])
        self.line.set_data(xdata_window[old_data_idx:], ydata_window[old_data_idx:])


        #self.line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()
        #print('Plot atualizado')
        time.sleep(0.001)

    def clear_plot(self):
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()

    def update_show_selected(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(REFRESH_RATE)  # interval in milliseconds

        # connect timer to update_data function
        self.timer.timeout.connect(self.show_selected_items)
        # start the timer
        self.timer.start()
    def update_subcomponent_list(self):
        global CURRENT_TOPIC
        global DF_MAIN_PLOT
        global CHANGE
        global SUBSCRIBE
        global CONNECT
        
        
        #print("TEST")
        selected_component = self.combobox.currentText()
        
        #print(CURRENT_TOPIC)
        print(selected_component)
        if CURRENT_TOPIC == None:
            CURRENT_TOPIC = selected_component
            print("0")
        try:
            app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            print("TEST")
            if CURRENT_TOPIC != selected_component:
                CHANGE=True
                self.client.disconnect()
                CONNECT = 0
                self.model.clear()
                self.client.connect(mqtt_broker_address, mqtt_broker_port)
                while CONNECT == 0:
                    print(CONNECT)
                    #time.sleep(0.1)
                DF_MAIN_PLOT = None
                SUBSCRIBE=0
            
            CURRENT_TOPIC = selected_component
            if SUBSCRIBE == 0:
                self.client.subscribe(CURRENT_TOPIC, qos=0)
                SUBSCRIBE+=1
                print("2")
            print("3")    
            
            # Start the MQTT client loop to receive incoming messages
            self.client.loop_start()
            time.sleep(2)
            subcomponents = Get_SubComponents_Names()
            
            # Limpando a lista de subcomponentes
            
            
            # Adicionando os subcomponentes ao modelo
            for i in range(0, len(subcomponents), 3):
                #print(subcomponents)
                subcomponent1 = subcomponents[i]
                subcomponent2 = subcomponents[i+1]
                subcomponent3 = subcomponents[i+2]
                item = QStandardItem(subcomponent1[:-2])
                # Adicionando sub-subcomponentes ao item
                subsubcomponents = []
                subsubcomponents.append(subcomponent1)
                subsubcomponents.append(subcomponent2)
                subsubcomponents.append(subcomponent3)
                
                for subsubcomponent in subsubcomponents:
                    subitem = QStandardItem(subsubcomponent)
                    item.appendRow(subitem)
                self.model.appendRow(item)
            
        except Exception as e:
            print(e)
            CURRENT_TOPIC = selected_component
            
            pass
        app.restoreOverrideCursor()
    
# Set the on_message callback function for the MQTT client
    def show_selected_items(self):
        
        global CURRENT_TOPIC
        global DF_MAIN_PLOT
        global CHANGE
        # Obtendo o item selecionado na lista de subcomponentes
        selected_component = self.combobox.currentText()
        ###########
        
            
            
    # Faça algo com o item da ComboBox
        ###########
        selected_indexes = self.tree.selectedIndexes()
        
        if selected_indexes:
            for x in selected_indexes:
                # Obtendo o texto do rótulo do item selecionado
                item_text = x.data()
                parent_parent = x.parent().data()
                mqtt_topic = selected_component
                #print(mqtt_topic)
                #time.sleep(1000)
                # Obtendo o texto do rótulo do pai do item selecionado
                parent_text = selected_component
                #print("Item: ", item_text)
                #print("Main: ", parent_text)
                df = None
                if item_text != CURRENT_TOPIC:
                    CURRENT_TOPIC = item_text
                    CHANGE = True
                if parent_parent:
                    #print("Parent: ", parent_parent)
                    df = get_df_column(parent_text,parent_parent,item_text)
                    model = CustomTableModel(df)
                    #app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
                    array = df.values.tolist()
                    self.update_plot(np.arange(len(array)),array)
                else:

                    self.clear_plot()
                    ########
                    arr = []
                    num_items = self.tree.model().rowCount()
                    for row in range(num_items):
                        item = self.tree.model().item(row)
                        # Check if this item has the name we're looking for
                        if item.text() == item_text:
                            # Iterate over all child items of this item and print their text
                            num_child_items = item.rowCount()
                            for child_row in range(num_child_items):
                                child_item = item.child(child_row)
                                
                                arr.append(child_item.text())
                            # Break out of the loop since we found the item we were looking for
                            break
                    ########
                    #for x in batat_widget:
                    #    print(x)
                    df = get_df_full(arr)
                    array = np.sqrt(np.sum(df.iloc[:, :3]**2, axis=1)).values.tolist()
                    self.update_plot(np.arange(len(array)),array)
                    model = CustomTableModel(df)
                    #app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
                 
                self.table.setModel(model)
                self.table.resizeColumnsToContents()
                self.table.setFont(QFont("Calibri", 10, QFont.Bold))
                #app.restoreOverrideCursor()



if __name__ == '__main__':
    #p = multiprocessing.Process(target=check_mqtt)
    #p.start()
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    window.setWindowIcon(QIcon(MAIN_DIR+"\\"+'icon.png'))
    window.setWindowTitle("Dashboard")
    sys.exit(app.exec_())