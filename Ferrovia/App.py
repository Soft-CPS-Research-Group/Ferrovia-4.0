from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QComboBox, QTreeView, QTableView, QVBoxLayout, QHBoxLayout, QAbstractItemView, QHeaderView
from PyQt5.QtGui import QStandardItemModel, QStandardItem,QCursor, QPixmap,QFont,QColor,QBrush,QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import QModelIndex, Qt,QAbstractTableModel,QAbstractItemModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import time
import ctypes

MAIN_DIR = "C:\\Users\\steam\\Desktop\\Dados Ferrovia"
REFRESH_RATE_SECS = 10
REFRESH_RATE = REFRESH_RATE_SECS * 1000
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


def Get_Components_Names():
    #cwd = os.getcwd()
    txt_files = [file.replace(".csv","") for file in os.listdir(MAIN_DIR) if file.endswith('.csv')]
    for file in txt_files:
        print(file)
    return txt_files

def Get_SubComponents_Names(file_name):
    df = pd.read_csv(MAIN_DIR+"\\"+file_name+'.csv')
    column_names = df.columns.tolist()
    #print(column_names)
    new_arr = column_names[0::3]
    return [s[:-2] for s in new_arr]

def get_df_column(parent_text,parent_parent,item_text):
    csv_name = MAIN_DIR+"\\"+parent_text+".csv"
    column_name = ""
    df = None
    try:
        column_name = parent_parent+"_"+item_text
        df = pd.read_csv(csv_name, usecols=[column_name])
    except:
        column_name = parent_parent+"_"+item_text.upper()
        df = pd.read_csv(csv_name, usecols=[column_name])
    #print(df)
    return df

def get_df_full(parent_text,parent_parent):
    cols = ["x","y","z"]
    csv_name = MAIN_DIR+"\\"+parent_text+".csv"
    column_name = ""
    df = None
    try:
        column_name = parent_parent+"_"
        df = pd.read_csv(csv_name, usecols=[column_name+cols[0],column_name+cols[1],column_name+cols[2]])
    except:
        column_name = parent_parent+"_"
        df = pd.read_csv(csv_name, usecols=[column_name+cols[0].upper(),column_name+cols[1].upper(),column_name+cols[2].upper()])
    #print(df)
    #print(df)
    return df

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Criando os componentes do combóio
        self.combobox = QComboBox()
        self.combobox.addItems(Get_Components_Names())

        # Add image 
        
        self.setStyleSheet("color: #333633;")
        pixmap = QPixmap(MAIN_DIR+"\\"+"train.png")
        #pixmap2 = QPixmap(MAIN_DIR+"\\"+"legend.png")
        label = QLabel(self)
        #label2 = QLabel(self)
        label.setPixmap(pixmap)
        
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Data Plot')
        
        # Criando a lista de subcomponentes
        self.model = QStandardItemModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers) # impedindo a edição dos itens
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents) # ajustando o tamanho das colunas
        self.tree.setExpandsOnDoubleClick(False) # impedindo que os itens se expandam ao serem clicados duas vezes
        
        self.tree.setHeaderHidden(True) # adicionando esta linha
        # Criando a tabela com os dados
        self.table = QTableView()
        
        # Adicionando os componentes ao layout
        vbox = QVBoxLayout()
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
        selection_model = self.tree.selectionModel()
        selection_model.selectionChanged.connect(self.show_selected_items)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(REFRESH_RATE)  # interval in milliseconds

        # connect timer to update_data function
        self.timer.timeout.connect(self.show_selected_items)

        # start the timer
        self.timer.start()

    def update_plot(self, x, y):
        # Atualiza os dados do gráfico e o redesenha
        self.line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()

        print('Plot atualizado')

    def clear_plot(self):
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()

    def update_subcomponent_list(self):
        # Simulando a busca de subcomponentes
        app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        self.model.clear()
        selected_component = self.combobox.currentText()

        subcomponents = Get_SubComponents_Names(selected_component)
        # Limpando a lista de subcomponentes
        
        
        # Adicionando os subcomponentes ao modelo
        for subcomponent in subcomponents:
            item = QStandardItem(subcomponent)
            # Adicionando sub-subcomponentes ao item
            subsubcomponents = ["x", "y", "z"]
            for subsubcomponent in subsubcomponents:
                subitem = QStandardItem(subsubcomponent)
                item.appendRow(subitem)
            self.model.appendRow(item)
        app.restoreOverrideCursor()

    def show_selected_items(self):
        # Obtendo o item selecionado na lista de subcomponentes
        selected_component = self.combobox.currentText()
        selected_indexes = self.tree.selectedIndexes()
        if selected_indexes:
            for x in selected_indexes:
                # Obtendo o texto do rótulo do item selecionado
                item_text = x.data()
                parent_parent = x.parent().data()
                # Obtendo o texto do rótulo do pai do item selecionado
                parent_text = selected_component
                #print("Item: ", item_text)
                #print("Main: ", parent_text)
                df = None
                if parent_parent:
                    #print("Parent: ", parent_parent)
                    df = get_df_column(parent_text,parent_parent,item_text)
                    model = CustomTableModel(df)
                    app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
                    array = df.values.tolist()
                    self.update_plot(np.arange(len(array)),array)
                else:
                    self.clear_plot()
                    df = get_df_full(parent_text,item_text)
                    array = np.sqrt(np.sum(df.iloc[:, :3]**2, axis=1)).values.tolist()
                    self.update_plot(np.arange(len(array)),array)
                    model = CustomTableModel(df)
                    app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
                 
                self.table.setModel(model)
                self.table.resizeColumnsToContents()
                self.table.setFont(QFont("Calibri", 10, QFont.Bold))
                app.restoreOverrideCursor()

                    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    window.setWindowIcon(QIcon(MAIN_DIR+"\\"+'icon.png'))
    window.setWindowTitle("Dashboard")
    sys.exit(app.exec_())