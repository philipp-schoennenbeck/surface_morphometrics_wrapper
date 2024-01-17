import mrcfile
from pathlib import Path
import utils
import mrc2xyz, xyz2ply, ply2vtp, curvature, intradistance_verticality, interdistance_orientation, thickness
import ray
import glob
import re
import shutil
import logging
import time
import click
from scipy.ndimage import label
import numpy as np
VERBOSE=False


def clean_up_segmentation(config, basenames):
    global VERBOSE
    original_dir = config["data_dir"] / "original_data"
    for mrc_file, basename in basenames.items():
        if VERBOSE:
            print(f"Cleaning up {mrc_file}")
        

        if config["clean_up_segmentation"]["save_original_data"]:
            original_dir.mkdir(exist_ok=True)
            new_path = original_dir / mrc_file.name
            if new_path.exists():
                print(f"Not copying {mrc_file} to {new_path} because it already exists.")
            else:
                shutil.copy(mrc_file, new_path)
        with mrcfile.open(mrc_file, "r+", permissive=True ) as f:
            data = f.data * 1
            new_data = np.zeros_like(data)
            all_labels = np.unique(data)
            for current_label in all_labels:
                if current_label == 0:
                    continue
                l, nl = label(data == current_label)
                uniques, unique_counts = np.unique(l, return_counts=True)
                for u, uc in zip(uniques, unique_counts):
                    if u == 0:
                        continue
                    if uc >= config["clean_up_segmentation"]["min_size"]:
                        new_data[l == u] = current_label
            f.set_data(new_data)






def run_surface_generation_per_file(config, output, seg_values, label, mrc_file, data):
    global VERBOSE
    xyz_file = Path(str(output) + ".xyz")
    ply_file = Path(str(output) + ".ply")
    vtp_file =  Path(str(output) + ".surface.vtp")
    ret_val = mrc2xyz.mrc_to_xyz(mrc_file, xyz_file, seg_values[label], config["surface_generation"]["angstroms"], data, verbose=VERBOSE) # Convert the segmentation file to xyz files
    if ret_val == 0:
        if VERBOSE:
            print(f"Generating a ply mesh with Screened Poisson: {ply_file}")
        ret_val = xyz2ply.xyz_to_ply(xyz_file, ply_file, 
                                        pointweight=config["surface_generation"]["point_weight"], 
                                        simplify=config["surface_generation"]["simplify"], 
                                        num_faces=config["surface_generation"]["max_triangles"], 
                                        k_neighbors=config["surface_generation"]["neighbor_count"], 
                                        deldist=config["surface_generation"]["extrapolation_distance"], 
                                        smooth_iter=config["surface_generation"]["smoothing_iterations"],
                                        depth=config["surface_generation"]["octree_depth"], verbose=VERBOSE)
        if ret_val != 0:
            if VERBOSE:
                print("Error converting xyz file to ply")
            return
        # Convert the ply file to a vtp file
        
        if VERBOSE:
            print(f"Converting the ply file to a vtp file: {vtp_file}")
        ply2vtp.ply_to_vtp(ply_file, vtp_file)


def run_surface_generation_per_file_cc(config, orig_output, seg_values, label, mrc_file, data, ind_dir):
    global VERBOSE
    basename = Path(orig_output).stem + "_" + label
    output = ind_dir / basename

    xyz_files = mrc2xyz.mrc_to_xyz_cc(mrc_file, output, seg_values[label], config["surface_generation"]["angstroms"], data, verbose=VERBOSE) # Convert the segmentation file to xyz files
    ply_files = []
    vtp_files = []
    for xyz_file in xyz_files:
        
        if xyz_file is not None:
            xyz_file = Path(xyz_file)
            ply_file = xyz_file.with_suffix(".ply")
            if VERBOSE:
                print(f"Generating a ply mesh with Screened Poisson: {ply_file}")
            
            ret_val = xyz2ply.xyz_to_ply(str(xyz_file), str(ply_file), 
                                            pointweight=config["surface_generation"]["point_weight"], 
                                            simplify=config["surface_generation"]["simplify"], 
                                            num_faces=config["surface_generation"]["max_triangles"], 
                                            k_neighbors=config["surface_generation"]["neighbor_count"], 
                                            deldist=config["surface_generation"]["extrapolation_distance"], 
                                            smooth_iter=config["surface_generation"]["smoothing_iterations"],
                                            depth=config["surface_generation"]["octree_depth"], verbose=VERBOSE)
            
            if ret_val is None:
                if VERBOSE:
                    print("Error converting xyz file to ply")
                continue
            # Convert the ply file to a vtp file
            vtp_file = Path(ply_file).with_suffix(".surface.vtp")
            if VERBOSE:
                print(f"Converting the ply file to a vtp file: {vtp_file}")
            ply2vtp.ply_to_vtp(str(ply_file), str(vtp_file))
            ply_files.append(ply_file)
            vtp_files.append(vtp_file)
            
    
    utils.combine_xyz_files(xyz_files, orig_output.parent / f"{basename}.xyz")
    utils.combine_ply_files(ply_files, orig_output.parent / f"{basename}.ply")
    utils.combine_vtp_files(vtp_files, orig_output.parent / f"{basename}.vtp")









@ray.remote
def run_curvature_on_file(surface, output_dir, config):
    global VERBOSE
    output_csv, output_gt, output_vtp = curvature.run_pycurv(surface, output_dir,
                            scale=1.0,
                            radius_hit=config["curvature_measurements"]["radius_hit"],
                            min_component=config["curvature_measurements"]["min_component"],
                            exclude_borders=config["curvature_measurements"]["exclude_borders"],
                            cores=config["curvature_measurements"]["pycurv_cores"], verbose=VERBOSE)
    return output_csv, output_gt, output_vtp 



@ray.remote
def combine_ind_files_after_pycurv_remote(avv_vtp_files, scaled_clean_vtp_files, surface_vtp_files, avv_gt_files, avv_csv_files, scaled_clean_gt_files, current_basename:Path, radius_hit):
    def get_order(files, search_str):
        order = []
        for file in files:
            result = re.search(search_str, str(file)).group(1)
            order.append(int(result))
        return order
    
    utils.combine_vtp_files(avv_vtp_files, current_basename.with_suffix(f".AVV_rh{radius_hit}.vtp"))
    utils.combine_vtp_files(scaled_clean_vtp_files, current_basename.with_suffix(f".scaled_cleaned.vtp"))
    utils.combine_vtp_files(surface_vtp_files, current_basename.with_suffix(f".surface.vtp"))
    utils.combine_gt_files(avv_gt_files, current_basename.with_suffix(f".AVV_rh{radius_hit}.gt"), get_order(avv_gt_files, f'.*{current_basename.name}_(.*).AVV_rh{radius_hit}.gt'))
    utils.combine_csv_files(avv_csv_files, current_basename.with_suffix(f".AVV_rh{radius_hit}.csv"), get_order(avv_csv_files, f'.*{current_basename.name}_(.*).AVV_rh{radius_hit}.csv'))
    utils.combine_gt_files(scaled_clean_gt_files, current_basename.with_suffix(f".scaled_cleaned.gt"), get_order(scaled_clean_gt_files, f'.*{current_basename.name}_(.*).scaled_cleaned.gt'))  

def combine_ind_files_after_pycurv(config, ind_dir, basenames):
    global VERBOSE
    def get_order(files, search_str):
        order = []
        for file in files:
            result = re.search(search_str, str(file)).group(1)
            order.append(int(result))
        return order


    radius_hit = config["curvature_measurements"]["radius_hit"]
    results = []
    for mrc_file, basename in basenames.items():
        for lab, current_basename in basename.items():
            if VERBOSE:
                print(f"Combining individual files for {current_basename.name}.")
            avv_label_basename = current_basename.name + f"_*.AVV_rh{radius_hit}"
            label_basename = current_basename.name + "_*"
            avv_csv_files = glob.glob(str(ind_dir/(avv_label_basename + ".csv")))
            avv_gt_files = glob.glob(str(ind_dir/(avv_label_basename + ".gt")))
            avv_vtp_files = glob.glob(str(ind_dir/(avv_label_basename + ".vtp")))
            scaled_clean_vtp_files = glob.glob(str(ind_dir/(label_basename + "scaled_cleaned.vtp")))
            scaled_clean_gt_files = glob.glob(str(ind_dir/(label_basename + "scaled_cleaned.gt")))
            surface_vtp_files = glob.glob(str(ind_dir/(label_basename + "surface.vtp")))
            current_basename:Path
            results.append(combine_ind_files_after_pycurv_remote.remote(avv_vtp_files, scaled_clean_vtp_files, surface_vtp_files, avv_gt_files, avv_csv_files, scaled_clean_gt_files, current_basename, radius_hit))
            
    results = [ray.get(res) for res in results]
            # utils.combine_vtp_files(avv_vtp_files, current_basename.with_suffix(f".AVV_rh{radius_hit}.vtp"))
            # utils.combine_vtp_files(scaled_clean_vtp_files, current_basename.with_suffix(f".scaled_cleaned.vtp"))
            # utils.combine_vtp_files(surface_vtp_files, current_basename.with_suffix(f".surface.vtp"))
            # utils.combine_gt_files(avv_gt_files, current_basename.with_suffix(f".AVV_rh{radius_hit}.gt"), get_order(avv_gt_files, f'.*{current_basename.name}_(.*).AVV_rh{radius_hit}.gt'))
            # utils.combine_csv_files(avv_csv_files, current_basename.with_suffix(f".AVV_rh{radius_hit}.csv"), get_order(avv_csv_files, f'.*{current_basename.name}_(.*).AVV_rh{radius_hit}.csv'))
            # utils.combine_gt_files(scaled_clean_gt_files, current_basename.with_suffix(f".scaled_cleaned.gt"), get_order(scaled_clean_gt_files, f'.*{current_basename.name}_(.*).scaled_cleaned.gt'))  
            

def run_curvature(config, ind_dir, basenames):
    global VERBOSE
    
    
    if config["separate_connected_components"]:
        mesh_files = glob.glob(str(ind_dir) +"/*.surface.vtp")
        output_dir = ind_dir
        
    else:

        mesh_files = glob.glob(str(config["work_dir"])+"/*.surface.vtp")
        output_dir = config["work_dir"]
    mesh_files = [Path(f) for f in mesh_files]

    results = []
    for surface in mesh_files:
        if VERBOSE:
            print("Processing "+str(surface))
        results.append(run_curvature_on_file.remote(surface, output_dir, config))
    results = [ray.get(res) for res in results]

    if config["separate_connected_components"]:
        combine_ind_files_after_pycurv(config, ind_dir, basenames)


@ray.remote
def intradistance_verticality_remote(dist_settings, graphname, surfacename):
    if dist_settings["verticality"]:
            intradistance_verticality.surface_verticality(str(graphname),verbose=VERBOSE)
    intradistance_verticality.surface_self_distances(str(graphname), str(surfacename),
                                                    dist_min=dist_settings["mindist"],
                                                    dist_max=dist_settings["maxdist"],
                                                    tolerance=dist_settings["tolerance"],
                                                    exportcsv=True,verbose=VERBOSE)
@ray.remote
def interdistance_orientation_remote(graphname1, current_label1, graphname2, current_label2, dist_settings):
    interdistance_orientation.surface_to_surface(str(graphname1), current_label1,
                                                str(graphname2), current_label2,
                                                orientation=dist_settings["relative_orientation"],
                                                save_neighbor_index=True,
                                                exportcsv=True, verbose=VERBOSE)

def run_distance_orientations(config, basenames):
    global VERBOSE
    radius_hit = config["curvature_measurements"]["radius_hit"]
    dist_settings = config["distance_and_orientation_measurements"]
    result = []
    for mrcfile, basename in basenames.items():
        for current_label, current_basename in basename.items():
            if dist_settings["intra"]:
                if current_label in dist_settings["intra"]:
                    current_basename:Path
                    graphname = current_basename.with_suffix(f".AVV_rh{radius_hit}.gt")
                    if not graphname.exists():
                        if VERBOSE:
                            print(f"No file found for {graphname.name}")
                            print("Skipping this label for this tomogram")
                        continue
                    if VERBOSE:
                        print(f"Intra-surface distances for {graphname.name}")
                    surfacename = graphname.with_suffix(".vtp")
                    result.append(intradistance_verticality_remote.remote(dist_settings, graphname, surfacename))
                    # if dist_settings["verticality"]:
                    #     intradistance_verticality.surface_verticality(str(graphname),verbose=VERBOSE)
                    # intradistance_verticality.surface_self_distances(str(graphname), str(surfacename),
                    #                                                 dist_min=dist_settings["mindist"],
                    #                                                 dist_max=dist_settings["maxdist"],
                    #                                                 tolerance=dist_settings["tolerance"],
                    #                                                 exportcsv=True,verbose=VERBOSE)
    # Inter-surface distances
        if dist_settings["inter"]:
            for current_label1, current_basename1 in basename.items():
                if current_label1 in dist_settings["inter"]:
                    comparison = dist_settings["inter"][current_label1]
            
                    graphname1 = current_basename1.with_suffix(f".AVV_rh{radius_hit}.gt")
                    if not graphname1.exists():
                        if VERBOSE:
                            print(f"No file found for {graphname1.name}")
                            print(f"Skipping all intersurface measurements for label {current_label1}")
                        continue
                    for current_label2, current_basename2 in basename.items():
                        if current_label2 in comparison:
                            graphname2 = current_basename2.with_suffix(f".AVV_rh{radius_hit}.gt")
                            if not graphname2.exists():
                                if VERBOSE:
                                    print(f"No file found for {graphname2.name}")
                                    print(f"Skipping comparison with {current_label2}")
                                continue
                            if VERBOSE:
                                print(f"Inter-surface distances for {current_label1} and {current_label2}")
                            result.append(interdistance_orientation_remote.remote(graphname1, current_label1, graphname2, current_label2, dist_settings))
                            # interdistance_orientation.surface_to_surface(str(graphname1), current_label1,
                            #                                                         str(graphname2), current_label2,
                            #                                                         orientation=dist_settings["relative_orientation"],
                            #                                                         save_neighbor_index=True,
                            #                                                         exportcsv=True, verbose=VERBOSE)
    [ray.get(res) for res in result]


    


def test_stuff():
    pass

def run_surface_generation(config, basenames, ind_dir, seg_values):
    global VERBOSE
    results = []
    mrc_ids = []
    for mrc_file, basename in basenames.items():
        mrc = mrcfile.open(mrc_file)
        
        # mrc_ids.append(ray.put(mrc.data))
        for label, output in basename.items():
            if config["separate_connected_components"]:
                run_surface_generation_per_file_cc(config, output, seg_values, label, mrc_file, mrc.data, ind_dir)
            else:
                results.append(run_surface_generation_per_file(config, output, seg_values, label, mrc_file, mrc.data))
    # results = [ray.get(res) for res in results]   
    del mrc_ids             



@click.command()
@click.argument('config', type=click.Path(exists=True, readable=True,file_okay=True, dir_okay=False, path_type=Path), )
def run(config):
    global VERBOSE
    config_file = config
    date = time.strftime("%Y%m%d-%H%M%S")

    config = utils.readConfig(config_file)
    basenames = utils.get_basenames(config)
    ind_dir = config["work_dir"] / "individual_files"
    log_dir = Path(config["work_dir"] / "logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=log_dir / f"{date}_warnings.txt", level=logging.DEBUG)
    logging.captureWarnings(True)
    # basenames = utils.get_basenames(config)
    seg_values = config["segmentation_values"]
    if "verbose" in config:
        VERBOSE=config["verbose"]
    if config["separate_connected_components"]:
        ind_dir.mkdir(parents=True, exist_ok=True)


    tmp_dir =Path.home() / "ray"
    ray.init( _system_config={ 'automatic_object_spilling_enabled':False }, num_cpus=config["cores"], _temp_dir=str(tmp_dir))
    session = Path(tmp_dir) / "session_latest"
    session = session.resolve()

    try:
        if config["clean_up_segmentation"]["run"]:
            if VERBOSE:
                print("Starting to run segmentation clean up")
            clean_up_segmentation(config, basenames)
        if config["surface_generation"]["run"]:
            if VERBOSE:
                print("Starting to run surface generation:")
            run_surface_generation(config, basenames, ind_dir, seg_values)
        if config["curvature_measurements"]["run"]:
            if VERBOSE:
                print("Starting to run curvature measurements")
            run_curvature(config, ind_dir, basenames)
        
        if config["distance_and_orientation_measurements"]["run"]:
            if VERBOSE:
                print("Starting to run distance and orientation measurements:")
            run_distance_orientations(config, basenames)
        if config["thickness_estimations"]["run"]:
            if VERBOSE:
                print("Starting to run thickness estimations:")
            thickness.thickness(config, basenames)
    except Exception as e:
        ray.shutdown()
        if session.exists():
            shutil.rmtree(session, ignore_errors=True)
        raise e
    ray.shutdown()
    if session.exists():
        shutil.rmtree(session, ignore_errors=True)




if __name__ == "__main__":
    run()
