import yaml
from pathlib import Path
import os
import mrcfile
from scipy.ndimage import label
import numpy as np
import pandas as pd
import pymeshlab as pm
from pycurv import TriangleGraph,pycurv_io
from graph_tool import load_graph
from graph_tool.generation import graph_union

def get_mrc_files(config):
    mrcfiles = []
    for file in os.listdir(config["data_dir"]):
        file = config["data_dir"] / file
        if file.suffix == ".mrc":
            mrcfiles.append(file)
    return mrcfiles

def get_basenames(config):
    basenames = {}
    mrcfiles = get_mrc_files(config)
    for file in mrcfiles:
        basenames[file] = get_basename(config, file)
    return basenames


def get_basename(config, mrc):
    basenames = {}
    with mrcfile.open(mrc, permissive=True) as f:
        data = f.data * 1
        for seg_label, seg_value in config["segmentation_values"].items():
            if seg_value in data:
                basenames[seg_label] = config["work_dir"] / (mrc.stem + f"_{seg_label}")
    return basenames

def fixConfig(config):
    config["data_dir"] = Path(config["data_dir"])
    config["work_dir"] = Path(config["work_dir"])
    config["work_dir"].mkdir(parents=True, exist_ok=True)
    return config

def readConfig(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return fixConfig(config)


def get_connected_components(data, lab):    
    l, num_features = label(data==lab, np.ones((3,3,3)))
    return l, num_features



def combine_xyz_files(files, output):
    df = None
    for file in files:
        current_df = pd.read_csv(file, sep=" ", header=None, index_col=False)
        if df is None:
            df = current_df
        else:
            df = pd.concat([df, current_df])
    if df is not None:
        df.to_csv(output, sep=" ", index=False, header=False)
    
def combine_vtp_files(files, output):  
    pycurv_io.merge_vtp_files([str(file) for file in files], output)


def combine_ply_files(files, output):
    ms = pm.MeshSet()
    for file in files:
        ms.load_new_mesh(str(file))
    ms.save_current_mesh(str(output))
    ms.clear()
    

def combine_gt_files(files, output, order):
    complete_graph = None
    properties = None
    
    for file,o in zip(files, order):
        if complete_graph is None:
            complete_graph = load_graph(file)
            properties = {"label":[np.ones(complete_graph.num_vertices()) * o]}
            value_types = {"label":"int32_t"}
            for prop_name in complete_graph.vertex_properties:
                property_ = np.array(list(getattr(complete_graph.vp, prop_name)))
                properties[prop_name] = [property_] 
                value_types[prop_name] = complete_graph.vp[prop_name].value_type()
        else:
            new_graph = load_graph(file)
            combined_graph = graph_union(complete_graph, new_graph)

            for prop_name in properties.keys():
                if prop_name == "label":
                    properties[prop_name].append(np.ones(new_graph.num_vertices()) * o)
                    continue
                property_ = np.array(list(getattr(new_graph.vp, prop_name)))
                properties[prop_name].append(property_)
        
            complete_graph = combined_graph
    if properties is None:
        if output is not None and complete_graph is not None:
        
            complete_graph.save(str(output))
        return
    for prop_name in properties.keys():
        setattr(complete_graph.vp, prop_name, complete_graph.new_vertex_property(value_types[prop_name], np.concatenate(properties[prop_name])))

        #  for prop_name in complete_graph.vertex_properties:
                
                
        #         first = np.array(list(getattr(complete_graph.vp, prop_name)))
        #         second = np.array(list(getattr(new_graph.vp, prop_name)))
                
        #         new_values = np.concatenate((first, second))

        #         setattr(combined_graph.vp,prop_name, combined_graph.new_vertex_property(complete_graph.vp[prop_name].value_type(), new_values ))
    
    if output is not None:
        
        complete_graph.save(str(output))


def combine_csv_files(files, output, order):
    df = None
    for file,o in zip(files, order):
        if df is None:
            df = pd.read_csv(file,index_col=0)
            df["label"] = o
        else:
            current_df = pd.read_csv(file,index_col=0)
            current_df["label"] = o
            df = pd.concat((df, current_df))
    if df is not None:
        df.to_csv(output)


if __name__ == "__main__":
    combine_xyz_files(["/Data/erc-3/schoennen/CET/membrain/SynPspA_EPl/test_cc/work_dir/individual_files/filtered_segmentation_mem_1_1.xyz",
                        "/Data/erc-3/schoennen/CET/membrain/SynPspA_EPl/test_cc/work_dir/individual_files/filtered_segmentation_mem_1_2.xyz"],
                        "/Data/erc-3/schoennen/CET/membrain/SynPspA_EPl/test_cc/test_output/output.xyz")