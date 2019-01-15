'''
Author: Luke Prince
Date: 11 January 2019
'''
import csv
import pandas as pd
import ephys
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

class Cell(object):
    def __init__(self, celldir):
        self.celldir  = celldir
        self.fluor    = pd.read_csv(celldir+'/Metadata_%s.csv'%celldir, nrows=1)['Celltype'].squeeze()
        self.layer    = pd.read_csv(celldir+'/Metadata_%s.csv'%celldir, nrows=1)['Layer'].squeeze()
        
    def extract_ephys(self, pulse_on, pulse_len, sampling_frequency, I_min, I_max, I_step):
        self.ephys   = ephys.Electrophysiology(celldir = self.celldir, 
                                               pulse_on = pulse_on, pulse_len = pulse_len,
                                               sampling_frequency = sampling_frequency,
                                               I_min = I_min, I_max = I_max, I_step = I_step)
        self.ephys.extract_features()
        self.ephys.plot(closefig=True)
        
        return self.ephys.results
    
class CellCollection(OrderedDict):
    def __init__(self, *args):
        OrderedDict.__init__(self, args)
        
    def __getitem__(self, key):
        
        if type(key) is int:
            key = OrderedDict.keys()[key]
            
        return OrderedDict.__getitem__(self, key)
    
    def __setitem__(self, cell):
        OrderedDict.__setitem__(self, cell.celldir, cell)
        
    def add(self, cell):
        self.__setitem__(cell)
                
    def collect_results(self):
        self.collect_metadata()
        self.collect_ephys()
                
    def collect_metadata(self):
        if not hasattr(self, 'metadata'):
            self.metadata = {'Fluorescence':[],'Layer':[]}
        
        for cellid, celln in self.items():
            self.metadata['Fluorescence'].append(celln.fluor)
            self.metadata['Layer'].append(celln.layer)
                
    def collect_ephys(self):
        if not hasattr(self, 'ephys'):
            self.ephys = {'Vrest (mV)': [],'Input Resistance (megaohm)': [],'Cell Capacitance (pF)': [],
                          'Rheobase (nA)':[],'fI slope (Hz/nA)':[],
                          'Adaptation Ratio':[],'Sag Amplitude (mV)':[],
                          'Spike Threshold (mV)':[],'Spike Amplitude (mV)':[],
                          'Spike Halfwidth (ms)':[],'Membrane Time Constant (ms)':[]}
        
        for cellid, celln in self.items():
            if hasattr(celln, 'ephys'):
                for key, val in celln.ephys.results.items():
                    try:
                        self.ephys[key+' (%s)'%(str(val.units).split(' ')[1])].append(val.item())
                    except AttributeError:
                        self.ephys[key].append(val)
                        
    def plot_ephys_summary(self):
        fig,axs=plt.subplots(nrows=4,ncols=3,figsize=(12,12),sharey=False)
        axs[-1][-1].axis('off');
        fluors=['NF', 'PV+', 'SST+']
        colors = ['#F5A623', '#4A90E2', '#7ED321']
        cols = self.df_ephys.columns[2:]
        
        for ix,col in enumerate(cols):
            plt.sca(np.ravel(axs)[ix])
            df_tmp = self.df_ephys[['Fluorescence', col]]
            plt.title(col, fontsize=14)
            
            for jx,fl in enumerate(fluors):
                vals = df_tmp[df_tmp['Fluorescence']==fl][col]
                plt.plot(jx + np.random.randn(len(vals))*0.1, vals, '.', ms=8, color=colors[jx], alpha=0.5)
                
                # Reduce range of plot with outliers
                if col=='Adaptation Ratio':
                    plt.ylim(0,11)
                    plt.plot(jx, vals.median(), 's', ms=10, color=colors[jx])
                else:
                    plt.plot(jx, vals.mean(), 's', ms=10, color=colors[jx])

            plt.xticks(range(3), fluors)
        fig.subplots_adjust(hspace=0.4)
        return fig