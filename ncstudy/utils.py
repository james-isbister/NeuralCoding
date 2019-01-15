import numpy as np
from neo import io
import os
import pdb

'''
ABF_IO.py

A simple library with functions for loading ABF files using the NEO library.
'''


def load_episodic(filename):
    
	'''
	load_episodic(filename)

	Loads episodic recordings from pClamp data in 'filename'.

	Returns the following:

	trace: a numpy array of size [t, n, c], where t is the number of samples per episode,
	n is the number of episodes (sweeps) and c is the number of channels.

	cinfo: a dictionary containing lists with the names and units of each of the channels, 
	keys are 'names' and 'units'.

	'''

	# open the file
	try:
		r = io.AxonIO(filename=filename)
	except IOError as e:
		print('Problem loading specified file')

	# read file into blocks
	bl = r.read_block(lazy=False,cascade=True)

	# read in the header info
	head = r.read_header()

	# determine the input channels and their info
	chans  = head['listADCInfo']
	nchans = len(chans)
	cinfo  = {'names' : [], 'units' : []}
	for c in chans:
		cinfo['names'].append(c['ADCChNames'])
		cinfo['units'].append(c['ADCChUnits'])

	# determine the number of sweeps and their length
	nsweeps  = np.size(bl.segments)
	nsamples = head['protocol']['lNumSamplesPerEpisode']/nchans

	# initialize an array to store the data
	trace = np.zeros((nsamples,nsweeps,nchans))

	# load the data into the traces
	bl = r.read_block(lazy=False,cascade=True)
	for c in range(nchans):
		for s in range(nsweeps):
			#pdb.set_trace()
			trace[:,[s],[c]] = bl.segments[s].analogsignals[c]



	return (trace, cinfo)

def merge_dicts(*dict_args):
    results = {}
    for d in dict_args:
        results.update(d)
    return results