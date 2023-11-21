#! /usr/bin/env python
"""Generate a scaled point cloud xyz file from a labelled mrc map

usage: mrc2xyz input.mrc output.xyz -l label_int"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import click
import glob
import mrcfile
import numpy as np
import pandas
from pathlib import Path
import utils

@click.command()
@click.argument('mrc_file', type=str)
@click.argument('output', type=str)
@click.option('-l', '--label', type=int, default=1, help="Label for feature of interest")
@click.option('-a','--angstrom', type=bool, default=False, help="Scale output in angstroms (default nm)")
def click_convert(mrc_file, output, label, angstrom):
	"""Wrapper function to convert a mrc file to an xyz file with click"""
	mrc_to_xyz(mrc_file, output, label, angstrom)

def mrc_to_xyz(mrc_file, output, label, angstrom, data=None, verbose=False):
	"""Extract a segmented feature of interest from a mrc file and output as an xyz-formatted point cloud
	
	Arguments:
		input {str} -- Input mrc file
		output {str} -- Output xyz file
		label {int} -- Label for feature of interest
		angstrom {bool} -- Scale output in angstroms (default nm)
	"""
	if data is None:
		mrc = mrcfile.mmap(mrc_file, mode="r+", permissive=True)
		data = mrc.data
	else:
		mrc = mrcfile.open(mrc_file, mode="r", permissive=True, header_only=True)
		
	if angstrom:
		voxel_size = mrc.voxel_size.x
		origin = mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z
	else:
		voxel_size = mrc.voxel_size.x/10 # nm
		origin = mrc.header.origin.x/10, mrc.header.origin.y/10, mrc.header.origin.z/10
	 
	data = np.where(data == label)
	if len(data[0]) == 0:
		if verbose:
			print("No data found for label {}".format(label))
		return 1
	df = pandas.DataFrame(data={'x': data[2], 'y': data[1], 'z': data[0]})
	df = df * voxel_size + origin
	df.to_csv(output, sep=" ", index=False, header=False)
	return 0


def mrc_to_xyz_cc(mrc_file, output, label, angstrom, data=None, verbose=False):
	"""Extract a segmented feature of interest from a mrc file and output as an xyz-formatted point cloud
	
	Arguments:
		input {str} -- Input mrc file
		output {str} -- Output xyz file
		label {int} -- Label for feature of interest
		angstrom {bool} -- Scale output in angstroms (default nm)
	"""
	if data is None:
		mrc = mrcfile.mmap(mrc_file, mode="r+", permissive=True)
		data = mrc.data
	else:
		mrc = mrcfile.open(mrc_file, mode="r", permissive=True, header_only=True)
		
	if angstrom:
		voxel_size = mrc.voxel_size.x
		origin = mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z
	else:
		voxel_size = mrc.voxel_size.x/10 # nm
		origin = mrc.header.origin.x/10, mrc.header.origin.y/10, mrc.header.origin.z/10
	 
	
	l, num_features = utils.get_connected_components(data, label)
	
	
	returns = []
	if num_features == 0:
		if verbose:
			print("No data found for label {}".format(label))
		returns.append(None)
	for lab in range(num_features):
		
		lab += 1
		current_output = f"{output}_{lab}.xyz"
		data = np.where(l == lab)

		df = pandas.DataFrame(data={'x': data[2], 'y': data[1], 'z': data[0]})
		df = df * voxel_size + origin
		df.to_csv(current_output, sep=" ", index=False, header=False)
		returns.append(current_output)
	return returns



def convert_mitochondria(input_filename):
	"""Convert mitochondrial segmentation from a mrc file to a xyz file.
	Convenience function to quickly output all relevant features without needing to use click.
	Formatting of segmentation must be done as follows: Map value 0: background, 1: OMM, 2: IMM, 3 (optional): ER

	Arguments:
	input_filename {str} -- Input mrc file
	"""
	labels = ["OMM", "IMM", "ER"]
	base = input_filename.split("_")[0]
	mrc = mrcfile.mmap(input_filename, mode="r", permissive=True)
	voxel_size = mrc.voxel_size.x/10 # nm
	# voxel_size = 1
	# print(voxel_size)
	for index, label in enumerate(labels):
		data = np.where(mrc.data == index+1)
		# print(data)
		df = pandas.DataFrame(data={'x': data[2], 'y': data[1], 'z': data[0]})
		df = df * voxel_size 
		output = base+"_{}.xyz".format(label)
		print(output)
		df.to_csv(output, sep=" ", index=False, header=False)

if __name__=="__main__":
	click_convert()