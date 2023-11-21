# Surface Morphometrics Pipeline Wrapper
Wrapper for the [Surface Morphometrics Pipeline](https://github.com/GrotjahnLab/surface_morphometrics). This wrapper aims to make the tools of the pipeline easier to use by adding inate support for parallel processing, adding support for separating connected components of the same label (the original mesh creation sometimes fails to create seperate meshes) as well as adding another analysis step for estimating the thickness of membranes.

To use these new features use the modified config file. Each pipeline step has a <em>run</em> flag to signal if this step is supposed to be running. Note that the steps are building on each other.

If the flag <em>separate_connected_components</em> is used, the pipeline will create separate files for each component of the same label (components which do not have neighbouring pixels in the segmentation) as well as create summed up versions for each label.

The <em>cores</em> parameter is now used to process most steps in parallel. Each label or connected component of a label creates a new process. 

Additionally there is another analysis step for estimating the thickness of membranes. At every point of the surface a cylinder perpendicular to the surface is cut out and the values along the normal at that point are averaged. From this plot with a prominent minimum (the dark pixels in the micrograph) the thickness can be estimated by various thresholds, gradients or neighbouring maxima. To use this part of the analysis pipeline an additional directory with the original micrograph has to be given as the <em>tomogram_dir</em> parameter.

To use this wrapper activate the python environment the usual way and afterwards run 
```bash
python run.py config.yaml
```

The normal way to run the pipeline should also still work.

## Surface Morphometrics Pipeline
![Workflow Figure](https://raw.githubusercontent.com/GrotjahnLab/surface_morphometrics/master/Workflow_title.png)
### Quantification of Membrane Surfaces Segmented from Cryo-ET or other volumetric imaging.  
Author: __Benjamin Barad__/*<benjamin.barad@gmail.com>*. 

Developed in close collaboration with Michaela Medina

A pipeline of tools to generate robust open mesh surfaces from voxel segmentations of biological membranes
using the Screened Poisson algorithm, calculate morphological features including curvature and membrane-membrane distance
using pycurv's vector voting framework, and tools to convert these morphological quantities into morphometric insights.


## Installation:
1. Clone this git repository: `git clone https://github.com/grotjahnlab/surface_morphometrics.git`
2. Install the conda environment: `conda env create -f environment.yml`
3. Activate the conda environment: `conda activate morphometrics_wrapper`
4. Install additional dependencies: `pip install -r pip_requirements.txt`

## Example data

There is tutorial data available in the `example_data` folder. Uncompress the tar file with:
```bash
cd example_data
tar -xzvf examples.tar.gz
```

There are two example datasets: `TE1.mrc` and `TF1.mrc`.
You can open them with `mrcfile`, like so:

```python
import mrcfile

with mrcfile.open('TE1.mrc', permissive=True) as mrc:
    print(mrc.data.shape)  # TE1.mrc has shape (312, 928, 960)
```

In some cases the file header may be non-standard (for example, mrc files exported from Amira software). In these cases, the `permissive=True` keyword argument is required, and you can ignore the warning that the file may be corrupt. All the surface morphometrics toolkit scripts will still run correctly.

## Running the configurable pipeline

Running the full pipeline on a 4 core laptop with the tutorial datasets takes about 8 hours (3 
for TE1, 5 for TF1), mostly in steps 3 and 4. With cluster parallelization, the full pipeline 
can run in 2 hours for as many tomograms as desired.

1. Edit the `config.yml` file for your specific project needs.
2. Run the surface reconstruction for all segmentations: `python segmentation_to_meshes.py config.yml`
3. Run pycurv for each surface (recommended to run individually in parallel with a cluster): `python 
run_pycurv.py config.yml ${i}.surface.vtp`

    You may see warnings aobut the curvature, this is normal and you do not need to worry.

4. Measure intra- and inter-surface distances and orientations (also best to run this one in parallel for each original segmentation): `python measure_distances_orientations.py config.yml ${i}.mrc`
5. Combine the results of the pycurv analysis into aggregate Experiments and generate statistics and plots. This requires some manual coding using the Experiment class and its associated methods in the `morphometrics_stats.py`. Everything is roughly organized around working with the CSVs in pandas dataframes. Running  `morphometrics_stats.py` as a script with the config file and a filename will output a pickle file with an assembled "experiment" object for all the tomos in the data folder. Reusing a pickle file will make your life way easier if you have dozens of tomograms to work with, but it doesn't save too much time with just the example data...

### Examples of generating statistics and plots:
* `python single_file_histogram.py filename.csv -n feature` will generate an area-weighted histogram for a feature of interest in a single tomogram. I am using a variant of this script to respond to reviews asking for more per-tomogram visualizations!
* `python single_file_2d.py filename.csv -n1 feature1 -n2 feature2` will generate a 2D histogram for 2 features of interest for a single surface.
* `mitochondria_statistics.py` shows analysis and comparison of multiple experiment objects for different sets of tomograms (grouped by treatment in this case). Every single plot and statistic in the preprint version of the paper gets generated by this script.


## Running individual steps without pipelining
Individual steps are available as click commands in the terminal, and as functions

1. Robust Mesh Generation
    1. `mrc2xyz.py` to prepare point clouds from voxel segmentation
    2. `xyz2ply.py` to perform screened poisson reconstruction and mask the surface
    3. `ply2vtp.py` to convert ply files to vtp files ready for pycurv
2. Surface Morphology Extraction
    1. `curvature.py` to run pycurv in an organized way on pregenerated surfaces
    2. `intradistance_verticality.py` to generate distance metrics and verticality measurements within a surface.
    3. `interdistance_orientation.py` to generate distance metrics and orientation measurements between surfaces.
    4. Outputs: gt graphs for further analysis, vtp files for paraview visualization, and CSV files for         pandas-based plotting and statistics
3. Morphometric Quantification - there is no click function for this, as the questions answered depend on the biological system of interest!
    1. `morphometrics_stats.py` is a set of classes and functions to generate graphs and statistics with pandas.
    2. [Paraview](https://www.paraview.org/) for 3D surface mapping of quantifications.

## File Descriptions:
* Files with.xyz extension are point clouds converted, in nm or angstrom scale. This is a flat text file with `X Y Z` coordinates in each line.
* Files with .ply extension are the surface meshes (in a binary format), which will be scaled in nm or angstrom scale, and work in many different softwares, including [Meshlab](https://www.meshlab.net/). 
* Files with surface.vtp extension are the same surface meshes in the [VTK](https://vtk.org/) format.
        * The .surface.vtp files are a less cross-compatible format, so you can't use them with as many types of software, but they are able to store all the fun quantifications you'll do!. [Paraview](https://www.paraview.org/) or [pyvista](https://docs.pyvista.org/) can load this format. This is the format pycurv reads to build graphs.
* Files with .gt extension are triangle graph files using the `graph-tool` python toolkit. These graphs enable rapid neighbor-wise operations such as tensor voting, but are not especially useful for manual inspection.
* Files with .csv extension are quantification outputs per-triangle. These are the files you'll use to generate statistics and plots.
* Files with .log extension are log files, mostly from the output of the pycurv run.
* Quantifications (plots and statistical tests) are output in csv, svg, and png formats. 

## Troubleshooting
0. If installation fails, you may want to reference `Install.md` for advanced installation tips (this is especially relevant for M1/M2 macs and Centos7 linux machines)
1. Warnings of the type `Gaussian or Mean curvature of X has a large computation error`... can be ignored, as they get cleaned up by pycurv
2. MRC files that are output by AMIRA don't have proper machine stamps by default. They need to be imported with `mrcfile.open(filename, permissive=True)` 

## Dependencies
1. Numpy
2. Scipy
3. Pandas
4. mrcfile
5. Click
6. Matplotlib
7. Pymeshlab
8. Pycurv   
    1. Pyto
    2. Graph-tool


## Citation
The development of this toolkit and examples of useful applications can be found in the following manuscript. Please cite it if you use this software in your research, or extend it to make improvements!

> **A surface morphometrics toolkit to quantify organellar membrane ultrastructure using cryo-electron tomography.**  
> Benjamin A. Barad<sup>†</sup>, Michaela Medina<sup>†</sup>, Daniel Fuentes, R. Luke Wiseman, Danielle A. Grotjahn  
> *bioRxiv* 2022.01.23.477440; doi: https://doi.org/10.1101/2022.01.23.477440

All scientific software is dependent on other libraries, but the surface morphometrics toolkit is particularly dependent on [PyCurv](https://github.com/kalemaria/pycurv), which provides the vector voted curvature measurements and the triangle graph framework. As such, please also cite the pycurv manuscript:

> **Reliable estimation of membrane curvature for cryo-electron tomography.**  
> Maria Salfer,Javier F. Collado,Wolfgang Baumeister,Rubén Fernández-Busnadiego,Antonio Martínez-Sánchez  
> *PLOS Comp Biol* August 2020; doi: https://doi.org/10.1371/journal.pcbi.1007962  

