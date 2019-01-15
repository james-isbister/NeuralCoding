'''
Author: Luke Prince
Date: 22 November 2018

Analysis of electrophysiological properties of neurons by current injection.

Traces are membrane potentials over the course of an injected current pulse of a known amplitude and duration.
'''

import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from quantities import pF, MOhm, nA, mV, ms, Hz
import matplotlib.pyplot as plt

from .cell import Cell
from .utils import load_episodic

class Electrophysiology(Cell):
    
    def __init__(self, celldir, pulse_on, pulse_len, sampling_frequency, I_min, I_max, I_step):
        super(Electrophysiology, self).__init__(celldir)
        self.pulse_on           = pulse_on
        self.pulse_off          = pulse_on + pulse_len
        self.pulse_len          = pulse_len
        self.pulse_off_ix       = int(self.pulse_off*sampling_frequency)
        self.pulse_on_ix        = int(self.pulse_on*sampling_frequency)
        self.sampling_frequency = sampling_frequency
        self.I_min              = I_min
        self.I_max              = I_max
        self.I_step             = I_step
        self.I_inj              = np.arange(I_min, I_max + I_step, I_step)
        
    #------------------------------------------------------------------
    #------------------------------------------------------------------

    def extract_features(self):
        self.data       = load_episodic(self.celldir+'/Ic_step.abf')[0][:,:,0]
        self.total_time = len(self.data)/self.sampling_frequency
        self.time       = np.arange(0, self.total_time, 1./self.sampling_frequency)

        '''True/False threshold crossing at each index'''
        threshCross = np.array([findThreshCross(trace) for trace in self.data.T])

        '''Count number of threshold crossings'''
        self.spikeCount = np.array([np.count_nonzero(tc) for tc in threshCross])

        '''Estimate Spike Frequency from number of threshold crossings'''
        self.spikeFreq = self.spikeCount*1000./self.pulse_len # in Hz

        '''Estimate resting Membrane Potential'''
        self.v_rest = resting_MP(self.data[:self.pulse_on_ix]) # mV

        '''Estimate Sag Amplitude'''
        self.v_sag = sag_amplitude(self.data[self.pulse_on_ix:self.pulse_off_ix,0]) # mV

        '''Input resistance measurement from traces where there was no threshold cross, and I>0'''
        self.Rin = np.mean([input_resistance(trace[self.pulse_on_ix:self.pulse_off_ix],I,self.v_rest) 
                            for (trace, I, crossed) in zip(self.data.T, self.I_inj, np.any(threshCross,axis=1))
                            if ((np.abs(I)>0) and not crossed)]) # MOhm

        '''Membrane time constant measurement from traces for 100ms after pulse off with no threshold cross, and I>0'''
        self.taum = np.mean([membrane_tau(trace[self.pulse_off_ix:self.pulse_off_ix+int(self.sampling_frequency*100)],
                                          self.sampling_frequency)
                             for (trace, I, crossed) in zip(self.data.T, self.I_inj, np.any(threshCross,axis=1))
                             if ((np.abs(I)>0) and not crossed)]) # ms

        '''Membrane capacitance calculation from Rin and taum'''
        self.Cm = 1000*self.taum/self.Rin # pF

        '''Estimate Slope of FI curve. Use I_inj at first non-zero spike count and initial increase as
           starting parameters for search. Use only max 5 points above rheobase to prevent attempting
           to fit to adapting FI curves.
        '''
        if self.spikeFreq.any():
            rheo_p0       = rheobase(self.spikeFreq, self.I_inj)
            slope_p0      = np.diff(np.nonzero(self.spikeFreq)[0])[0]/self.I_step

            rheo, fISlope = estimatefISlope(self.spikeFreq, self.I_inj, p0=[rheo_p0,slope_p0])

            self.rheobase = rheo
            self.fISlope  = fISlope
        else:
            self.rheobase = np.nan
            self.fISlope = np.nan

        '''Get spike times from threshold crossings'''
        self.spikeTimes = [self.time[1:][tc] for tc in threshCross]

        '''Estimate adaptation ratio if there are at least three spikes in a sweep'''
        self.adaptation_ratio = adaptation_ratio(self.spikeTimes[-1])

        '''Indices for spike times'''
        spikeix = [(sp*self.sampling_frequency).astype('int') for sp in self.spikeTimes]

        '''For each spike...'''
        ''' TODO: Make spike window selection smarter'''
        v_thresh = []
        v_amp = []
        v_ahp = []
        spikeHalfWidth = []
        for sp,trace in zip(spikeix,self.data.T):
            if np.any(sp):
                for ix in sp:
                    spikeTrace = trace[ix-29:ix+30] # Extract spike
                    v_thresh.append(spike_threshold(spikeTrace[:30])) # estimate spike threshold
                    v_amp.append(spike_amplitude(spikeTrace, v_thresh[-1])) # estimate spike amplitude
                    v_ahp.append(ahp_amplitude(spikeTrace, v_thresh[-1])) # estimate ahp amplitude
                    spikeHalfWidth.append(spike_halfwidth(spikeTrace, v_thresh[-1], v_amp[-1], self.sampling_frequency)) # estimate spike halfwidth

        self.v_thresh = np.mean(v_thresh)
        self.v_amp = np.mean(v_amp)
        self.v_ahp = np.mean(v_ahp)
        self.spikeHalfWidth = np.mean(spikeHalfWidth)

        self.results = {'Vrest': self.v_rest*mV,'Input Resistance': self.Rin*MOhm,'Cell Capacitance': self.Cm*pF,
                        'Rheobase':self.rheobase*nA,'fI slope':self.fISlope*Hz/nA, #'AHP amplitude' : self.ahpAmp*mV,
                        'Adaptation Ratio':self.adaptation_ratio,'Sag Amplitude':self.v_sag*mV,
                        'Spike Threshold':self.v_thresh*mV,'Spike Amplitude':self.v_amp*mV,
                        'Spike Halfwidth':self.spikeHalfWidth*ms,'Membrane Time Constant':self.taum*ms}
        
    #------------------------------------------------------------------
    #------------------------------------------------------------------

    def plot(self, num_traces=3, fig_height=4, include_fI = True, include_results = True, savefig = True, closefig=False):

        width_ratio = 3;
        if include_fI or include_results:
            include_ratio = ((width_ratio, True), (1, include_fI), (1, include_results))
            width_ratio = tuple([width for width,inc in include_ratio if inc])

            fig_width = fig_height * 4. * np.sum(width_ratio)/5.
            fig, axs = plt.subplots(ncols=len(width_ratio), figsize=(fig_width, fig_height), gridspec_kw={'width_ratios' : width_ratio})

            plt.sca(axs[0])

        else:
            fig_width = fig_height * 2.4
            fig = plt.figure(figsize=(fig_width, fig_height))

        idxs = np.array([i in np.percentile(self.I_inj,
                                            np.linspace(0, 100, num_traces),
                                            interpolation='nearest')
                         for i in self.I_inj])
        
        for ii,trace in enumerate(self.data.T[idxs]):
            plt.plot(self.time, trace, color = plt.cm.plasma(np.linspace(0, 1, num_traces)[ii]), lw=1, zorder=-ii)
            
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend(['I = %.2f'%I for I in self.I_inj[idxs]])
        
        plt.title('Cell ID: %s'%self.celldir+' %s'%self.fluor +' %s'%self.layer,fontsize=20)
        
        if include_fI:
            plt.sca(axs[1])
            plt.plot(self.I_inj, self.spikeFreq, 'o', ms=10)
            I_fine = np.linspace(self.I_min-self.I_step, self.I_max+self.I_step, 1000)
            plt.plot(I_fine, fICurve(I_fine, self.rheobase, self.fISlope), color='rebeccapurple', zorder=-1)
            plt.xlabel('I$\mathrm{_{inj}} (nA)$')
            plt.ylabel('Spike Frequency (Hz)')
            
        if include_results:
            plt.sca(axs[-1])
            plt.axis('off')
            Y = np.linspace(0.05,0.9,len(self.results.keys()))
            for ix,key in enumerate(self.results.keys()):
                plt.text(x = 0.1,y = Y[ix],s = key+' = ' + str(np.round(self.results[key], 2)),
                         transform=axs[-1].axes.transAxes, fontsize=14)
                
        if savefig:
            fig.savefig(self.celldir+'/ephys_features.svg')
            
        if closefig:
            plt.close(fig)
    
#------------------------------------------------------------------
#------------------------------------------------------------------

def steadyStateVoltage(trace,start_ix,end_ix):
    '''
    Calculate the steady state voltage of a slice between two timepoints of a trace

    Arguments:
        - trace (np.array) : array containing voltage trace in mV
        - start_ix (float)     : start time index to calculates steady state
        - end_ix (float)       : end time index to calculate steady state

    Returns:
        - v_ss (np.float)  : steady state voltage in mV of trace between t1 and t2
    '''

    # Calculate mean voltage in slice and return
    return np.mean(trace[start_ix:end_ix])

#------------------------------------------------------------------
#------------------------------------------------------------------

def sag_amplitude(trace):
    '''
    Calculate amplitude of the sag potential in response to a hyperpolarising current injection

    Argument:
        - trace (np.array) : array containing voltage trace in mV with hyperpolarising current injection

    Returns:
        - v_sag (np.float) : sag amplitude in mV

    '''
    # Calculate baseline voltage after current injection
    Vss = steadyStateVoltage(trace,-100,-1)

    # Find lowest voltage in first 1000 time points of trace
    Vmin = np.min(trace[:1000])

    # Return absolute difference between baseline and minimum membrane potential
    return np.abs(Vmin-Vss)

#------------------------------------------------------------------
#------------------------------------------------------------------


def adaptation_ratio(spikeTimes):
    '''
    Calculate adaptation ratio defined asratio between late and early inter-spike intervals in a train).
    Requires a minimum of 3 spikes.

    Arguments:
        - spikeTimes (list) : list of spike times in ms

    Return:
        - AR (float) : adaptation ratio
    '''

    # Set ratio to nan (in the event of insufficient spikes to calculate ratio
    AR = np.nan

    # For spike train with 3-6 spikes, calculate ratio between first and last inter-spike interval
    if len(spikeTimes)>2 and len(spikeTimes)<7:
        # Calculate inter-spike intervals as difference between spike times
        isi = np.diff(spikeTimes)
        # Divide last interval by first interval to obtain adaptation ratio
        AR = isi[-1]/isi[0]

    # For spike train with 7+ spikes, calculate ratio between average of first two and last two 
    # inter-spike intervals
    elif len(spikeTimes)>=7:
        # Calculate inter-spike intervals as difference between spike times
        isi = np.diff(spikeTimes)
        # Divide mean of last two intervals by mean of first two intervals to obtain adaptation ratio
        AR = np.mean(isi[-2:])/np.mean(isi[:2])

    # Return adaptation ratio
    return AR

#------------------------------------------------------------------
#------------------------------------------------------------------

def findThreshCross(trace,thresh=-20):
    '''
    Find threshold crosses of a trace

    Arguments:
        - trace (np.array) : membrane potential trace in mV in response to current injection
        - thresh (float)   : threshold to cross (default = -20)

    Returns:
        - bool_idx (np.array) : threshold crossings returned as a boolean index of length (len(trace)-2).
    '''

    # Determine time points greater than or equal to threshold, and time points less than threshold and take
    # union shifted by one time point
    return np.logical_and(trace[1:]>=thresh,trace[:-1]<thresh)

#------------------------------------------------------------------
#------------------------------------------------------------------

def resting_MP(trace):
    '''
    Estimate resting membrane potential

    Arguments:
        - trace (np.array) : membrane potential trace in mV in response to current injection

    Returns:
        - v_rest (np.float) : resting membrane potential in mV
    '''

    # Estimate resting membrane potential as average voltage in first 200 time points
    return steadyStateVoltage(trace,0,2000)

#------------------------------------------------------------------
#------------------------------------------------------------------

def fICurve(I,thresh,slope):
    '''
    Function to describe relationship between current injection in nA (I) and firing rate (f) in Hz of a neuron.
    A basic f-I curve assumes a relationship whereby the frequency increases linearly with current with a defined
    slope after a threshold is reached. Subthreshold current induces no firing (0 Hz).

    Arguments:
        - I (np.array)      : current in nA over which to define f-I curve
        - thresh (np.float) : threshold current in nA to induce non-zero firing rate
        - slope (np.float)  : slope in Hz/nA to define super-threshold frequency current relationship

    Returns:
        - f (np.array)      : firing rate in Hz
    '''

    return np.clip(slope*I - slope*thresh,a_min=0,a_max=np.inf)

#------------------------------------------------------------------
#------------------------------------------------------------------

def estimatefISlope(f,I,p0=[100,10], num_nonzero=5):
    '''
    Fit parameters (threshold and slope) of f-I curve to firing rate in Hz of sampled neurons
    with current injection values I. Uses curve_fit from scipy.optimize

    Arguments:
        - f (np.array)      : observed firing rates in Hz of neurons in response to current injection
        - I (np.array)      : current injection values in nA
        - p0 (list)         : initial parameter values (default=[100,10])
        - num_nonzero (int) : number of non-zero points to fit up to (default = 5)

    Returns:
        - [fitted_thresh, fitted_slope]
    '''
    ix = int(len(f) - np.maximum(np.count_nonzero(f) - num_nonzero, 0))
    popt,pcov = curve_fit(fICurve,I[:ix],f[:ix],p0=p0)
    return popt
#------------------------------------------------------------------
#------------------------------------------------------------------

def rheobase(f,I):
    '''
    Estimate rheobase from observed firing rate (f) and current injection values (I).
    Returns first current value for which firing rate is greater than zero

    Arguments:
        - f (np.array) : observed firing rates in Hz of neurons in response to current injection
        - I (np.array) : current injection values in nA

    Returns:
        - rheobase (np.float)
    '''

    return I[f>0][0]

#------------------------------------------------------------------
#------------------------------------------------------------------

def expDecay(t,V0,tau,offset):
    '''
    Function describing exponential decay from V0+offset to V0

    Arguments:
        - t (np.array)       : time-steps in ms
        - V0 (np.float)      : baseline voltage in mV
        - tau (np.float)     : time constant in ms
        - offset (np.float)  : offset voltage in mV

    Returns: 
        - v_decay (np.array) : exponentially decaying voltage
    '''

    return (V0-offset)*np.exp(-t/tau) + offset

#------------------------------------------------------------------
#------------------------------------------------------------------

def membrane_tau(trace, sampling_frequency):
    '''
    Fit membrane time constant of a trace.

    Arguments:
        - trace (np.array) : trace of exponentially decaying membrane potential

    Returns:
        - fitted_tau (np.float) : fitted time constant
    '''
    # Create array of time-steps in ms
    t = np.arange(0,len(trace)/sampling_frequency,1./sampling_frequency) # in ms

    # Fit time constant using curve_fit from scipy.optimize
    popt,pcov = curve_fit(expDecay,t,trace,p0=[trace[0],1.0,trace[-1]])
    return popt[1]

#------------------------------------------------------------------
#------------------------------------------------------------------

def input_resistance(trace,I,vrest):
    '''
    Estimate input resistance using Ohm's Law (V = IR)

    R_in    =     Change in steady state voltage due to current injection
                  _______________________________________________________

                                    Injected Current

    Arguments:
        - trace (np.array) : trace of membrane potential in mV
        - I (np.float)     : injected current in nA
        - vrest (np.float) : resting membrane potential in mV

    Returns:
        - Rin (np.float) : input resistance in MOhm
    '''
    vss = steadyStateVoltage(trace,-100,-1)
    return (vss-vrest)/I

#------------------------------------------------------------------
#------------------------------------------------------------------

def spike_threshold(spikeTrace,zslopeThresh=0.5):
    '''
    Estimate threshold membrane potential in mV of action potential by finding voltage at which membrane potential
    slope drastically increases

    Arguments:
        - spikeTrace (np.array)   : trace of action potential in mV
        - zslopeThresh (np.float) : z-scored membrane potential first derivative threshold

    Returns: 
        - v_thresh (np.float) : threshold membrane potential in mV
    '''

    # Calculate z-score of action potential slope
    zslope = stats.zscore(np.diff(spikeTrace))
    # find slope z-score threshold crossings 
    ap_thresh = [np.logical_and(zslope[1:]>=zslopeThresh,zslope[:-1]<zslopeThresh)]

    # Return first membrane potential crossing slope z-score threshold
    return spikeTrace[1:-1][ap_thresh][0]

#------------------------------------------------------------------
#------------------------------------------------------------------

def spike_amplitude(spikeTrace,Vthresh):
    '''
    Estimate action potential amplitude in mV given an estimated action potential threshold

    Arguments:
        - spikeTrace (np.array) : trace of action potential in mV
        - Vthresh (np.float) : action potential threshold

    Return:
        - v_ap (np.float) : action potential amplitude in mV
    '''

    Vpeak = np.max(spikeTrace)
    return Vpeak - Vthresh

#------------------------------------------------------------------
#------------------------------------------------------------------

def ahp_amplitude(spikeTrace,Vthresh):
    '''
    Estimate after-hyperpolarization amplitude in mV given an estimated action potential threshold

    Arguments:
        - spikeTrace (np.array) : trace of action potential in mV
        - Vthresh (np.float) : action potential threshold

    Return:
        - v_ahp (np.float) : after-hyperpolarization amplitude in mV

    '''
    Vtrough = np.min(spikeTrace)
    return Vthresh - Vtrough

#------------------------------------------------------------------
#------------------------------------------------------------------

def spike_halfwidth(spikeTrace, Vthresh, spikeAmplitude, sampling_frequency):
    '''
    Estimate spike half-width in ms given an estimate of action potential threshold and spike amplitude.
    Spike half-width is the duration above mid-voltage of the action potential
    '''

    # time-steps of action potential trace
    time = np.arange(0,len(spikeTrace)/sampling_frequency,1./sampling_frequency)

    # Calculate mid-voltage
    Vhalf = Vthresh + spikeAmplitude/2.

    # Find time points above mid-voltage
    time = time[spikeTrace>=Vhalf]

    # Return difference between first and last time points above mid-voltage + additional time step for correction
    return time[-1] - time[0] + 1./sampling_frequency
    