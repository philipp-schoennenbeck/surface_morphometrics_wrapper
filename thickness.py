import numpy as np
from matplotlib import pyplot as plt
import mrcfile
from skimage.draw import disk
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from scipy import signal
from scipy.ndimage import map_coordinates
from shapely.geometry import LineString # to install
from fast_histogram import histogram1d # to install
from datetime import datetime
from scipy.interpolate import splrep, BSpline
import yaml
from sys import argv
import shutil

import glob, os
os.environ["RAY_DEDUP_LOGS"] = "0"
# np.seterr(all='raise')

import ray
import ray.cloudpickle as cloudpickle
pd.options.mode.chained_assignment = None






def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def matlab_style_gauss3D(shape=(3,3,3),sigma=0.5):
    import numpy as np
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n,o = [(ss-1.)/2. for ss in shape]
    y,x,z = np.ogrid[-m:m+1,-n:n+1,-o:o+1]
    h = np.exp( -(x*x + y*y + z*z) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def thickness_by_median_threshold(profile, median_value, pixel_size, max_distance=100, method=None, gradient_cutoff=0.002, output_dict=None, crop=10, ignore_restrictions=False):
    def plot_unusable(why,one, two=None, three=None):
        if output_dict is None or output_dict["output_path"] is None:
            return
        if why not in output_dict:
            output_dict[why] = 0
        if output_dict["max_output"] <= output_dict[why]:
            return
        
        counter = output_dict[why]
        output_dict[why] += 1
        process = output_dict["process"]
        idx = output_dict["idx"]
        if three is not None:
            fig, ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
            ax[0].plot(one)
            ax[0].hlines([median_value], [0], [len(one) - 1 ], colors="red")
            ax[1].plot(two[0])
            ax[0].set_title("Profile")
            ax[1].set_title("Profile gradient")
            ax[2].set_title("Cropped profile gradient")
            ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
            ax[2].hlines([gradient_cutoff,-gradient_cutoff], [0,0], [len(three)-1,len(three)-1], colors="red")
            plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {median_value}\nCropping at: {two[1][0]}, {two[1][1]}, {two[1][2]}, {two[1][3]}\n Gradient cutoff: {gradient_cutoff}")
            
            ax[2].scatter(np.arange(len(three)), three)

        elif two is not None:
            fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
            ax[0].plot(one)
            ax[0].hlines([median_value], [0], [len(one) - 1 ], colors="red")
            ax[0].set_title("Profile")
            ax[1].set_title("Profile gradient")
            ax[1].plot(two[0])
            ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
            plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {median_value}\nCropping at: {two[1][0]} - {two[1][1]} and {two[1][2]} - {two[1][3]}")
            
        else:
            plt.figure(constrained_layout=True)
            plt.plot(one)
            plt.hlines([median_value], [0], [len(one) - 1 ], colors="red")
            plt.title(f"Method: {method}\nReason: {why}\nMedian: {median_value}")
            plt.hlines([median_value], [0], [len(one)])
            
        plt.savefig(Path(output_dict["output_path"]) / f"{process}_{method}_{why}_{counter}_{idx}.png")
        plt.close()
    x_ = np.linspace(0,len(profile),len(profile),endpoint=False)
    first_line = LineString(np.column_stack((x_, profile)))
    second_line = LineString(np.column_stack((x_, np.ones(len(profile)) * median_value)))
    intersection = first_line.intersection(second_line)

    thickness = np.nan
    reason = "No intersection"
    if intersection.geom_type == 'MultiPoint':
        xs = []
        ys = []
        for s in list(intersection.geoms):
            xs.append(s.x)
            ys.append(s.y)
        
        argsorted_xs = np.argsort(xs).astype(int)
        xs = np.array(xs)
        ys = np.array(ys)
        xs = xs[argsorted_xs]
        ys = ys[argsorted_xs]
        
        minima,_ = signal.find_peaks(profile * -1)
        if not ignore_restrictions:
            middle = len(profile) // 2
            
            minima_distance = [abs(middle - minimum) for minimum in minima]
            usable_minima = [minimum for minimum, minimum_distance in zip(minima, minima_distance) if minimum_distance < max_distance]
            
            if len(usable_minima) == 0:
                plot_unusable("No usable minimum", profile)
                return np.nan, "No usable minimum"
            lowest_minimum = usable_minima[np.argmin([profile[i] for i in usable_minima])]
        else:
            lowest_minimum = np.argmin(profile)
        maxima,_ = signal.find_peaks(profile)
        
        for p in range(len(maxima) -1 ):
            if maxima[p] < lowest_minimum and maxima[p+1] > lowest_minimum :
                lower_max = maxima[p]
                upper_max = maxima[p+1]
                break
        else:
            if not ignore_restrictions:
                plot_unusable("No maxima", profile)
                return np.nan, "No maxima"


        for p in range(len(xs) -1 ):
            if xs[p] < lowest_minimum and xs[p+1] > lowest_minimum:
                if ignore_restrictions or (xs[p] >= lower_max and xs[p+1] <= upper_max):
                    thickness = (xs[p+1] - xs[p] ) # * pixel_size
                    reason = "Worked"
                    break
                break
        else:
            
            plot_unusable("Bad intersection", profile)
            return np.nan, "Bad intersection"
        if np.logical_not(np.isnan(thickness)):
            real_gradient = np.gradient(profile)
            # crop = 10
            # a,b,c,d = np.floor(xs[p] + crop).astype(np.int32), lowest_minimum - crop, lowest_minimum + crop, np.ceil(xs[p+1] - crop).astype(np.int32)
            a,b,c,d = np.floor(xs[p]).astype(np.int32), lowest_minimum - crop, lowest_minimum + crop, np.ceil(xs[p+1]).astype(np.int32)
            if a > b or c > d:
                if not ignore_restrictions:
                    thickness = np.nan
                    reason = "Bad cropping around minimum"
                    plot_unusable(reason, profile, (real_gradient, (a,b,c,d)))
            else:
                gradients = real_gradient[a:b+1]
                gradients = np.concatenate((gradients, real_gradient[c:d+1]))
                if gradient_cutoff is not None and np.any(np.abs(gradients) < gradient_cutoff):
                    if not ignore_restrictions:
                        thickness = np.nan
                        reason = "Below gradient cutoff"
                        plot_unusable(reason, profile, (real_gradient, (a,b,c,d)), gradients)

    else:
        plot_unusable(reason, profile)

    return thickness, reason

def thickness_by_minimum_between_maxima(profile, threshold,max_distance=100, gradient_cutoff=0.002, output_dict=None, method="Between_maxima", crop=15, ignore_restrictions=False):
    def plot_usable(one, two, three, idx):
         
        fig, ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
        # ax[0].set_title()
        ax[0].plot(one)
        # ax[0].hlines([threshold], [0], [len(one) - 1 ], colors="red")
        ax[1].plot(two)
        ax[0].set_title("Profile")
        ax[1].set_title("Profile gradient")
        ax[2].set_title("Cropped profile gradient")
        # ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
        # ax[2].hlines([gradient_cutoff,-gradient_cutoff], [0,0], [len(three)-1,len(three)-1], colors="red")

        # plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {threshold}\nCropping at: {two[1][0]}, {two[1][1]}, {two[1][2]}, {two[1][3]}\n Gradient cutoff: {gradient_cutoff}")
        
        ax[2].scatter(np.arange(len(three)), three)
        plt.savefig(Path(output_dict["output_path"]) / f"{idx}_{method}.png")
        plt.close()
    
    def plot_unusable(why,one, two=None, three=None):
        if output_dict is None or output_dict["output_path"] is None:
            return
        if why not in output_dict:
            output_dict[why] = 0
        if output_dict["max_output"] <= output_dict[why]:
            return
        
        counter = output_dict[why]
        output_dict[why] += 1
        process = output_dict["process"]
        idx = output_dict["idx"]
        if three is not None:
            fig, ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
            ax[0].plot(one)
            ax[0].hlines([threshold], [0], [len(one) - 1 ], colors="red")
            ax[1].plot(two[0])
            ax[0].set_title("Profile")
            ax[1].set_title("Profile gradient")
            ax[2].set_title("Cropped profile gradient")
            ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
            ax[2].hlines([gradient_cutoff,-gradient_cutoff], [0,0], [len(three)-1,len(three)-1], colors="red")

            plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {threshold}\nCropping at: {two[1][0]}, {two[1][1]}, {two[1][2]}, {two[1][3]}\n Gradient cutoff: {gradient_cutoff}")
            
            ax[2].scatter(np.arange(len(three)), three)

        elif two is not None:
            fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
            ax[0].plot(one)
            ax[0].hlines([threshold], [0], [len(one) - 1 ], colors="red")
            ax[0].set_title("Profile")
            ax[1].set_title("Profile gradient")
            ax[1].plot(two[0])
            ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
            plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {threshold}\nCropping at: {two[1][0]} - {two[1][1]} and {two[1][2]} - {two[1][3]}")
            
        else:
            plt.figure(constrained_layout=True)
            plt.plot(one)
            plt.hlines([threshold], [0], [len(one) - 1 ], colors="red")
            plt.title(f"Method: {method}\nReason: {why}\nMedian: {threshold}")
            plt.hlines([threshold], [0], [len(one)])
            
        plt.savefig(Path(output_dict["output_path"]) / f"{process}_{method}_{why}_{counter}_{idx}.png")
        plt.close()

    x_ = np.linspace(0,len(profile),len(profile),endpoint=False)
    first_line = LineString(np.column_stack((x_, profile)))
    second_line = LineString(np.column_stack((x_, np.ones(len(profile)) * threshold)))
    intersection = first_line.intersection(second_line)

    thickness = np.nan
    reason = "No intersection"
    if intersection.geom_type == 'MultiPoint' or ignore_restrictions:
        if not ignore_restrictions:
            xs = []
            ys = []
            for s in list(intersection.geoms):
                xs.append(s.x)
                ys.append(s.y)
            
            argsorted_xs = np.argsort(xs).astype(int)
            xs = np.array(xs)
            ys = np.array(ys)
            xs = xs[argsorted_xs]
            ys = ys[argsorted_xs]    




        maxima,_ = signal.find_peaks(profile)
        if not ignore_restrictions:
            minima,_ = signal.find_peaks(profile * -1)
            middle = len(profile) // 2
            
            minima_distance = [abs(middle - minimum) for minimum in minima]
            usable_minima = [minimum for minimum, minimum_distance in zip(minima, minima_distance) if minimum_distance < max_distance]
            if len(usable_minima) == 0 or len(maxima) < 2:
                plot_unusable("No usable minimum", profile)
                return np.nan, "No usable minimum"
            lowest_minimum = usable_minima[np.argmin([profile[i] for i in usable_minima])]
        else:
            lowest_minimum = np.argmin(profile)
    
        thickness = np.nan
        for p in range(len(maxima) -1 ):
            if maxima[p] < lowest_minimum and maxima[p+1] > lowest_minimum :
                thickness = (maxima[p+1] - maxima[p] )
                reason = "Worked"
                break
        if not ignore_restrictions:
            for r in range(len(xs)-1):
                if xs[r] > maxima[p] and xs[r] < lowest_minimum and xs[r+1] < maxima[p+1] and xs[r+1] > lowest_minimum:
                    break
            else:
                thickness = np.nan
                reason = "Intersection above maxima"
                plot_unusable(reason, profile)

        if np.logical_not(np.isnan(thickness)):
            real_gradient = np.gradient(profile)
            # crop = 15
            a,b,c,d = np.floor(maxima[p] + crop).astype(np.int32), lowest_minimum - crop, lowest_minimum + crop, np.ceil(maxima[p+1] - crop).astype(np.int32)
            if (a > b or c > d) and not ignore_restrictions:
                thickness = np.nan
                reason = "Bad cropping around minimum"
                plot_unusable(reason, profile, (real_gradient, (a,b,c,d)))

            else:
                gradients = real_gradient[a:b+1]
                gradients = np.concatenate((gradients, real_gradient[c:d+1]))
                if (gradient_cutoff is not None and np.any(np.abs(gradients) < gradient_cutoff)) and not ignore_restrictions:
                    thickness = np.nan
                    reason = "Gradient below threshold"
                    plot_unusable(reason, profile, (real_gradient, (a,b,c,d)), gradients)
                elif output_dict["idx"] % 100 == 0:
                    plot_usable(profile, real_gradient, gradients, output_dict["idx"])

            

    return thickness, reason


def thickness_by_maximum_gradient(profile, threshold,max_distance=100, gradient_cutoff=0.002, output_dict=None, method="Gradient", crop=15, ignore_restrictions=False):
    def plot_usable(one, two, three, idx, four):
         
        fig, ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
        # ax[0].set_title()
        ax[0].plot(one)
        # ax[0].hlines([threshold], [0], [len(one) - 1 ], colors="red")
        ax[1].plot(two)
        ax[0].set_title("Profile")
        ax[1].set_title("Profile gradient")
        ax[2].set_title("Cropped profile gradient")
        # ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
        # ax[2].hlines([gradient_cutoff,-gradient_cutoff], [0,0], [len(three)-1,len(three)-1], colors="red")

        # plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {threshold}\nCropping at: {two[1][0]}, {two[1][1]}, {two[1][2]}, {two[1][3]}\n Gradient cutoff: {gradient_cutoff}")
        
        ax[2].scatter(np.arange(len(three)), three)
        ax[0].plot(four,[profile[int(four[0])],profile[int(four[1])]])
        plt.savefig(Path(output_dict["output_path"]) / f"{idx}_{method}.png")
        plt.close()
    
    def plot_unusable(why,one, two=None, three=None):
        if output_dict is None or output_dict["output_path"] is None:
            return
        if why not in output_dict:
            output_dict[why] = 0
        if output_dict["max_output"] <= output_dict[why]:
            return
        
        counter = output_dict[why]
        output_dict[why] += 1
        process = output_dict["process"]
        idx = output_dict["idx"]
        if three is not None:
            fig, ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
            ax[0].plot(one)
            ax[0].hlines([threshold], [0], [len(one) - 1 ], colors="red")
            ax[1].plot(two[0])
            ax[0].set_title("Profile")
            ax[1].set_title("Profile gradient")
            ax[2].set_title("Cropped profile gradient")
            ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
            ax[2].hlines([gradient_cutoff,-gradient_cutoff], [0,0], [len(three)-1,len(three)-1], colors="red")

            plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {threshold}\nCropping at: {two[1][0]}, {two[1][1]}, {two[1][2]}, {two[1][3]}\n Gradient cutoff: {gradient_cutoff}")
            
            ax[2].scatter(np.arange(len(three)), three)

        elif two is not None:
            fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
            ax[0].plot(one)
            ax[0].hlines([threshold], [0], [len(one) - 1 ], colors="red")
            ax[0].set_title("Profile")
            ax[1].set_title("Profile gradient")
            ax[1].plot(two[0])
            ax[1].vlines(two[1], [np.min(two[0]),np.min(two[0]),np.min(two[0]),np.min(two[0])], [np.max(two[0]),np.max(two[0]),np.max(two[0]),np.max(two[0])], colors="red")
            plt.suptitle(f"Method: {method}\nReason: {why}\nMedian: {threshold}\nCropping at: {two[1][0]} - {two[1][1]} and {two[1][2]} - {two[1][3]}")
            
        else:
            plt.figure(constrained_layout=True)
            plt.plot(one)
            plt.hlines([threshold], [0], [len(one) - 1 ], colors="red")
            plt.title(f"Method: {method}\nReason: {why}\nMedian: {threshold}")
            plt.hlines([threshold], [0], [len(one)])
            
        plt.savefig(Path(output_dict["output_path"]) / f"{process}_{method}_{why}_{counter}_{idx}.png")
        plt.close()

    x_ = np.linspace(0,len(profile),len(profile),endpoint=False)
    first_line = LineString(np.column_stack((x_, profile)))
    second_line = LineString(np.column_stack((x_, np.ones(len(profile)) * threshold)))
    intersection = first_line.intersection(second_line)

    thickness = np.nan
    reason = "No intersection"
    if intersection.geom_type == 'MultiPoint' or ignore_restrictions:
        if not ignore_restrictions:
            xs = []
            ys = []
            for s in list(intersection.geoms):
                xs.append(s.x)
                ys.append(s.y)
            
            argsorted_xs = np.argsort(xs).astype(int)
            xs = np.array(xs)
            ys = np.array(ys)
            xs = xs[argsorted_xs]
            ys = ys[argsorted_xs]    




        maxima,_ = signal.find_peaks(profile)
        if not ignore_restrictions:
            minima,_ = signal.find_peaks(profile * -1)
            middle = len(profile) // 2
            
            minima_distance = [abs(middle - minimum) for minimum in minima]
            usable_minima = [minimum for minimum, minimum_distance in zip(minima, minima_distance) if minimum_distance < max_distance]
            if len(usable_minima) == 0 or len(maxima) < 2:
                plot_unusable("No usable minimum", profile)
                return np.nan, "No usable minimum"
            lowest_minimum = usable_minima[np.argmin([profile[i] for i in usable_minima])]
        else:
            lowest_minimum = np.argmin(profile)
    
        thickness = np.nan
        for p in range(len(maxima) -1 ):
            if maxima[p] < lowest_minimum and maxima[p+1] > lowest_minimum :
                thickness = (maxima[p+1] - maxima[p] )
                reason = "Worked"
                break
        if not ignore_restrictions:
            for r in range(len(xs)-1):
                if xs[r] > maxima[p] and xs[r] < lowest_minimum and xs[r+1] < maxima[p+1] and xs[r+1] > lowest_minimum:
                    break
            else:
                thickness = np.nan
                reason = "Intersection above maxima"
                plot_unusable(reason, profile)

        if np.logical_not(np.isnan(thickness)):
            real_gradient = np.gradient(profile)


            start = maxima[p]
            end = maxima[p+1]
            lowest_gradient = np.argmin(real_gradient[start:lowest_minimum+1]) + start
            highest_gradient = np.argmax(real_gradient[lowest_minimum:end+1]) + lowest_minimum
            thickness = highest_gradient - lowest_gradient
            
            if not ignore_restrictions:

                a,b,c,d = lowest_gradient, lowest_minimum - crop, lowest_minimum + crop, highest_gradient
                if a > b or c > d:
                    thickness = np.nan
                    reason = "Bad cropping around minimum"
                    plot_unusable(reason, profile, (real_gradient, (a,b,c,d)))

                else:
                    gradients = real_gradient[a:b+1]
                    gradients = np.concatenate((gradients, real_gradient[c:d+1]))
                    if gradient_cutoff is not None and np.any(np.abs(gradients) < gradient_cutoff):
                        thickness = np.nan
                        reason = "Gradient below threshold"
                        plot_unusable(reason, profile, (real_gradient, (a,b,c,d)), gradients)
                    else:
                        if output_dict["idx"] % 50 == 0 or thickness > 80:
                            plot_usable(profile, real_gradient, gradients, output_dict["idx"], (lowest_gradient, highest_gradient))
                
                    
                    
            

    return thickness, reason






def estimate_thickness(tomo_data, tomo_seg, normals, points, radius=45, height=400, pixel_size=1, points_to_calc=1000, return_thickness_map=True, methods=["global_median"],
                       profile_window_size=10, profile_sigma=1.5, max_output=10, process_id=None, output_path=None, threshold_gradient_crop=10, maxima_gradient_crop=15, gradient_threshold=0.0015, reduce_thresholds=0,
                       ignore_restrictions=False):
    def clip(value, axis=0):
        return np.clip(value, 0, tomo_data.shape[axis]-1)
    MAIN_VECTOR = (0,0,1)
    counter = 0
    idx_counter = 0
    radius = int(radius / pixel_size)
    height = int(height / pixel_size)
    neighbourhood_distance = int(1000 / pixel_size)
    if return_thickness_map:

        thickness_map = {method:np.zeros_like(tomo_data) for method in methods}

    x,y = disk((0,0), radius * 3)
    x = np.array(x, dtype=float) / 3 
    y = np.array(y, dtype=float) / 3
    z = np.ones_like(x)

    median_value = np.median(tomo_data)
    
    gradient_cutoff = gradient_threshold
    total_coordinates = []
    local_medians = []
    local_modes = []
    coordinate_counts = []
    total_counts = []
    total_points = []

    output_dict = {"max_output":max_output, "process":process_id, "idx":0, "output_path":output_path}

    thicknesses = []
    usable = []
    reasons = {method:{"unusable":0} for method in methods}
    for point_counter, (normal_vector, point) in enumerate(zip(normals, points)):
        counter += 1
        if np.any(point <height // 2) or np.any(point > (np.array(tomo_data.shape) - height // 2)) or np.isclose(np.sum(normal_vector + MAIN_VECTOR),0):
            usable.append(False)
            coordinate_counts.append([])
            local_medians.append(0)
            total_counts.append(0)
            local_modes.append(0)
            total_coordinates.append(np.empty((3,0)))
            total_points.append(None)
            for method in methods:
                reasons[method]["unusable"] += 1
        else:

                
            
            
            try:
                mat = rotation_matrix_from_vectors(MAIN_VECTOR, normal_vector)
            except Exception as e:
                print(MAIN_VECTOR, normal_vector)
                raise e

            values = []
            interpolated_values = []
            coordinates = []
            coordinate_count = []

            for i in range(height):
                disk_coords = np.array((x,y,z * (i - height // 2)) )
                rotated = mat.dot(disk_coords)

                
                for i in range(3):
                    rotated[i] += point[i]


                x_,y_,z_ = rotated.astype(int)
                
                usable_idxs = np.where((x_ >= 0) & (x_ < tomo_data.shape[0]) & (y_ >= 0) & (y_ < tomo_data.shape[1]) & (z_ >= 0) & (z_ < tomo_data.shape[2]))
                x_ = x_[usable_idxs]
                y_ = y_[usable_idxs]
                z_ = z_[usable_idxs]
                if len(usable_idxs[0]) == 0:
                    break

                current_slice = np.mean(tomo_data[x_,y_,z_])
                values.append(current_slice)
                coordinates.append(rotated)
                coordinate_count.append(len(x_))
            if len(coordinates) == 0:
                usable.append(False)
                coordinate_counts.append([])
                local_medians.append(0)
                total_counts.append(0)
                local_modes.append(0)
                total_coordinates.append(np.empty((3,0)))
                total_points.append(None)
                for method in methods:
                    reasons[method]["unusable"] += 1
                continue
            

            local_neighbourhood = tomo_data[clip(point[0] - neighbourhood_distance//2,0) :clip(point[0] + neighbourhood_distance//2,0),clip(point[1] - neighbourhood_distance//2,1):clip(point[1] + neighbourhood_distance//2,1),clip(point[2] - neighbourhood_distance//2,2):clip(point[2] + neighbourhood_distance//2,2)]
            local_seg_neighbourhood = tomo_seg[clip(point[0] - neighbourhood_distance//2,0) :clip(point[0] + neighbourhood_distance//2,0),clip(point[1] - neighbourhood_distance//2,1):clip(point[1] + neighbourhood_distance//2,1),clip(point[2] - neighbourhood_distance//2,2):clip(point[2] + neighbourhood_distance//2,2)]
            try:
                local_median = np.median(local_neighbourhood[local_seg_neighbourhood == 0])
            except Exception as e:
                print(len(local_neighbourhood[local_seg_neighbourhood == 0]), np.unique(local_neighbourhood[local_seg_neighbourhood == 0], return_counts=True))
                print(len(local_neighbourhood[local_seg_neighbourhood != 0]), np.unique(local_neighbourhood[local_seg_neighbourhood != 0], return_counts=True))
                print(pixel_size, point, tomo_data.shape, neighbourhood_distance)
                raise e
            local_medians.append(local_median)
            total_points.append(point)

            # nr, boxes = np.histogram(local_neighbourhood[local_seg_neighbourhood == 0].flatten(), 50)
            cropped = local_neighbourhood[local_seg_neighbourhood == 0].flatten()
            cropped_min = np.min(cropped)
            cropped_max = np.max(cropped)
            nr = histogram1d(cropped.flatten(), 50, (cropped_min,cropped_max))
            boxes = np.linspace(cropped_min, cropped_max, 51)


            best_bin = np.argmax(nr)
            best_bin_value = (boxes[best_bin] + boxes[best_bin+1]) /2
            local_modes.append(best_bin_value)
            
        


            coordinates = np.concatenate(coordinates, -1)
            total_coordinates.append(coordinates)
            coordinate_counts.append(coordinate_count)
            total_counts.append(len(coordinates))
            usable.append(True)

            
        
        if counter % points_to_calc == 0 or point_counter == len(points) - 1:
            idx_counter += 1
            interpolated_values = map_coordinates(tomo_data, np.concatenate(total_coordinates,-1))

            current=0
            for current_counter, (use, coordinate_count, local_median, total_count, coordinates, point, local_mode) in enumerate(zip(usable, coordinate_counts, local_medians, total_counts, total_coordinates, total_points, local_modes)):
                if not use:
                    thicknesses.append(np.ones(len(methods)) * np.nan)
                    continue
                result = []
                for c in coordinate_count:
                    try:
                        result.append(np.mean(interpolated_values[current:current+c]))
                    except Exception as e:

                        
                        raise e
                    current += c
                # current += total_count


                # for window_size in range(1,len(orig_profile)//4):
                window = signal.gaussian(profile_window_size, profile_sigma)
                
                for c, v in enumerate([ result]): #values

                    
                    padded_profile = np.concatenate([np.ones((profile_window_size - 1) // 2)* v[0], v, np.ones((profile_window_size - 1) // 2)* v[-1]])

                    profile = signal.convolve(padded_profile, window/window.sum(), "valid")
                    nr_of_spline_points = 450
        
                    t,c,k = splrep(np.arange(len(profile)), profile)
                    profile = BSpline(t,c,k)(np.linspace(0, len(profile), nr_of_spline_points))

                    current_thickness_estimations = []
                    output_dict["idx"] = current_counter
                    # if current_counter == 0:
                    #     print(median_value, local_median, local_mode, idx_counter, len(points))
                    for method in methods:
                        if method =="global_median":
                            threshold = median_value - (median_value - np.min(profile)) * reduce_thresholds
                            thickness, reason = thickness_by_median_threshold(profile, threshold, pixel_size, method="global_median", gradient_cutoff=gradient_cutoff,output_dict=output_dict,crop=threshold_gradient_crop, ignore_restrictions=ignore_restrictions)
                        elif method == "local_median":
                            threshold = local_median - (local_median - np.min(profile)) * reduce_thresholds
                            thickness, reason = thickness_by_median_threshold(profile, threshold, pixel_size, method="local_median", gradient_cutoff=gradient_cutoff, output_dict=output_dict, crop=threshold_gradient_crop, ignore_restrictions=ignore_restrictions)
                        elif method == "local_mode":
                            threshold = local_mode - (local_mode - np.min(profile)) * reduce_thresholds
                            thickness, reason = thickness_by_median_threshold(profile, threshold, pixel_size, method="local_mode", gradient_cutoff=gradient_cutoff, output_dict=output_dict, crop=threshold_gradient_crop, ignore_restrictions=ignore_restrictions)
                        elif method == "maxima":
                            threshold = local_median - (local_median - np.min(profile)) * reduce_thresholds
                            thickness, reason = thickness_by_minimum_between_maxima(profile, threshold, gradient_cutoff=gradient_cutoff, output_dict=output_dict, crop=maxima_gradient_crop, ignore_restrictions=ignore_restrictions)
                        elif method == "gradient":
                            threshold = local_median - (local_median - np.min(profile)) * reduce_thresholds
                            thickness, reason = thickness_by_maximum_gradient(profile, threshold, gradient_cutoff=gradient_cutoff, output_dict=output_dict, crop=maxima_gradient_crop, ignore_restrictions=ignore_restrictions)
                        else:
                            raise NotImplementedError(f"Method \"{method}\" for thickness estimation is not implemented.")
                        if return_thickness_map:
                            if np.logical_not(np.isnan(thickness)):
                                try:
                                    thickness_map[method][point[0],point[1],point[2]] = thickness
                                except Exception as e:
                                    # print(method, type(thickness_map[method]), thickness, point, type(point))
                                    # for i in [usable, coordinate_counts, local_medians, total_counts, total_coordinates, total_points]:
                                    #     print(len(i))
                                    raise e
                        if reason not in reasons[method]:
                            reasons[method][reason] = 0
                        reasons[method][reason] += 1
                        current_thickness_estimations.append(thickness)
                    thicknesses.append(current_thickness_estimations)


            local_medians = []
            total_coordinates = []
            coordinate_counts = []
            coordinates = []
            total_counts = []
            usable = []
            total_points = []
            local_modes = []

    if return_thickness_map:
        return thicknesses, thickness_map, reasons
    return thicknesses, reasons




def gauss(fx,fy,fz,sig):

    r = np.fft.fftshift(np.sqrt(fx**2 + fy**2 + fz**2))
    res = -2*np.pi**2*(r*sig)**2

    return np.exp(res.astype(np.float32))

def gaussian_filter(im,sig,apix):
    '''
        sig (real space) and apix in angstrom
    '''
    sig = sig/2/np.pi


    fx,fy,fz = np.meshgrid(np.fft.fftfreq(im.shape[1],apix),\
                            np.fft.fftfreq(im.shape[0],apix),\
                            np.fft.fftfreq(im.shape[2],apix))

    im_fft = np.fft.fftshift(np.fft.fftn(im))
    fil = gauss(fx,fy, fz, sig*apix)
    # with mrcfile.new("/Data/erc-3/schoennen/membrane_analysis_toolkit/test_code/thickness_estimation/plots/gaus.mrc", data=fil.astype(np.float32), overwrite=True) as f:
    #     f.voxel_size = 10.9
    # print(fx.shape, fy.shape, im_fft.shape, fil.shape, im.shape)    
    im_fft_filtered = im_fft*fil
    newim = np.real(np.fft.ifftn(np.fft.ifftshift(im_fft_filtered)))
    
    return newim

def print_size(obj):
    serialized_data = cloudpickle.dumps(obj)
    data_size = len(serialized_data)
    return data_size



def find_tomograms(config, basenames):
    mrc_files = sorted(list(basenames.keys()))
    tomo_files = sorted(glob.glob(str(config["thickness_estimations"]["tomogram_dir"]) + "/*.mrc"))
    assert len(mrc_files) == len(tomo_files)
    tomograms = {mrc:tomo for mrc, tomo in zip(mrc_files, tomo_files)}
    return tomograms


estimate_thickness_remote = ray.remote(estimate_thickness)


def thickness(config, basenames):
    tomograms = find_tomograms(config, basenames)

    # Get some parameters from config
    cylinder_radius = config["thickness_estimations"]["cylinder_radius"]
    cylinder_height = config["thickness_estimations"]["cylinder_height"]
    profile_window_size = config["thickness_estimations"]["profile_window_size"]
    profile_sigma = config["thickness_estimations"]["profile_sigma"]
    output_path = Path(config["work_dir"])
    check_output_path = output_path / "check_output"
    # check_output_path = config["thickness_estimations"]["check_output_path"]
    check_output = config["thickness_estimations"]["check_output"]
    threshold_gradient_crop = config["thickness_estimations"]["threshold_gradient_crop"]
    maxima_gradient_crop = config["thickness_estimations"]["maxima_gradient_crop"]
    high_pass = config["thickness_estimations"]["high_pass_filter"]
    methods = config["thickness_estimations"]["methods"]
    gradient_threshold = config["thickness_estimations"]["gradient_threshold"]
    njobs = config["cores"]
    use_ray = True
    out_methods = [f"thickness_{method}" for method in methods]
    radius_hit = config["curvature_measurements"]["radius_hit"]
    

    # Set up ray if using it
    

    # Create an output dir to check the profiles
    if check_output > 0:
        if not Path(check_output_path).exists():
            Path(check_output_path).mkdir(parents=True)


    for mrc, basename in basenames.items():
        
    # for seg_file, tomo_file in zip(segmentation_files, tomograms):
        tomo_file = tomograms[mrc]
        # tomo_file = Path(config["thickness_estimations"]["tomogram_dir"]) / tomo_file
        tomo_data_orig = mrcfile.open(tomo_file, permissive=True)

        if config["thickness_estimations"]["pixel_size"] == 0:
            ps = tomo_data_orig.voxel_size["x"]
        else:
            ps = config["thickness_estimations"]["pixel_size"]

        if config["surface_generation"]["angstroms"]:
            conversion = ps 
        else:
            conversion = ps / 10

        tomo_data_orig = tomo_data_orig.data  * 1


        if config["thickness_estimations"]["clip"]:
            mean = np.mean(tomo_data_orig)
            std = np.std(tomo_data_orig)
            clip_stds = config["thickness_estimations"]["clip_stds"]
            tomo_data = np.clip(tomo_data_orig, mean-std*clip_stds,mean+std*clip_stds)
        else:
            tomo_data = tomo_data_orig
        del tomo_data_orig

        tomo_data -= np.mean(tomo_data)
        tomo_data /= np.std(tomo_data)

        if high_pass > 0:
            sig = int(high_pass / ps)
            sig += (sig + 1) % 2
            tomo_data = gaussian_filter(tomo_data,0,ps) - gaussian_filter(tomo_data,sig,ps)

        

        
        
        # csv_files = glob.glob(config["work_dir"] +  str(Path(seg_file).stem) +"*AVV_rh*.csv")
        # csv_files = sorted(csv_files)
        # csv_files = [csv_files[2]]
        # csv_files = csv_files[:1]

        with mrcfile.open(mrc, permissive=True) as tomo_seg:
            tomo_seg = tomo_seg.data * 1
        


        if config["thickness_estimations"]["smooth"]:
            smooth_kernel = config["thickness_estimations"]["smooth_kernel"]
            gaus = matlab_style_gauss3D((smooth_kernel,smooth_kernel,smooth_kernel),smooth_kernel)

            tomo_data = signal.convolve(tomo_data, gaus, mode="same")
        if use_ray:
            ray.put(tomo_data)
            ray.put(tomo_seg)


        for label, current_basename in basename.items():
            csv_file = current_basename.with_suffix(f".AVV_rh{radius_hit}.csv")
            # csv_file = config["work_dir"] + csv_file 

            

            if config["verbose"]:
                print(f"Estimating thickness for {current_basename.name}")
            if not csv_file.exists():
                if config["verbose"]:
                    print(f"{csv_file} does not exist. Run surface generation and pycurv before.")
                continue
            orig_csv = pd.read_csv(csv_file, sep=",")
            if all([method in orig_csv.columns for method in out_methods]) and not config["thickness_estimations"]["reestimate"]:
                if config["verbose"]:
                    print(f"Skipping thickness estimation for {current_basename.name} because results already exist and reestimate is flagged.")
                continue
            csv = orig_csv.copy()
            if not all(coord in csv.columns for coord in ["xyz_x", "xyz_y", "xyz_z", "normal_x","normal_y","normal_z"]):
                if config["verbose"]:
                    print(f"Needed columns in {csv_file} do not exist. Run pycurv before. ")
                continue
            csv["xyz_x_pixel"] = csv["xyz_x"] / conversion
            csv["xyz_y_pixel"] = csv["xyz_y"] / conversion
            csv["xyz_z_pixel"] = csv["xyz_z"] / conversion

            csv = csv.astype({"xyz_x_pixel":int,"xyz_y_pixel":int,"xyz_z_pixel":int})

            filtered_csv = csv.drop_duplicates(subset=["xyz_x_pixel","xyz_y_pixel","xyz_z_pixel"])


            x = np.array(filtered_csv["xyz_x_pixel"], dtype=int)
            y = np.array(filtered_csv["xyz_y_pixel"], dtype=int)
            z = np.array(filtered_csv["xyz_z_pixel"], dtype=int)

            normals = np.array((filtered_csv["normal_z"],filtered_csv["normal_y"],filtered_csv["normal_x"])).T
            points = np.array([z,y,x]).T

            current_check_output_path = Path(check_output_path) / Path(mrc).stem / label
            if not current_check_output_path.exists():
                current_check_output_path.mkdir(parents=True)
            if njobs > 1:
                if use_ray:
                    points_per_njob = int(len(points) / njobs) + 1
                    points = [points[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)]
                    normals = [normals[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)] 

                    
                    result = [estimate_thickness_remote.remote(tomo_data, tomo_seg, normal, point, cylinder_radius, cylinder_height, ps, 1000, True, 
                                                                        methods,profile_window_size, profile_sigma,
                                                                        check_output,process_id, current_check_output_path, threshold_gradient_crop, maxima_gradient_crop ) 
                                                                        for process_id, (point, normal) in enumerate(zip(points, normals))]
                    thickness_maps = {}
                    thicknesses = []
                    total_reasons = {}
                    for res in result:
                        th, th_map, reasons = ray.get(res)
                        for method, reas in reasons.items():
                            if method not in total_reasons:
                                total_reasons[method] = {}
                            for key, number in reas.items():
                                if key not in total_reasons[method]:
                                    total_reasons[method][key] = 0
                                total_reasons[method][key] += number

                        for method, thickness_map in th_map.items():
                            if method not in thickness_maps:
                                thickness_maps[method] = np.copy(thickness_map)
                            else:
                                thickness_maps[method] += np.copy(thickness_map)
                        thicknesses.extend(th)
                else:
                    with mp.Pool(njobs) as pool:
                        points_per_njob = int(len(points) / njobs) + 1
                        points = [points[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)]
                        normals = [normals[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)] 

                        result = [pool.apply_async(estimate_thickness, args=[tomo_data, tomo_seg,  normal, point, cylinder_radius, cylinder_height, ps, 1000, True, 
                                                                            methods,profile_window_size, profile_sigma,
                                                                            check_output,process_id, current_check_output_path, threshold_gradient_crop, maxima_gradient_crop ]) 
                                                                            for process_id, (point, normal) in enumerate(zip(points, normals))]
                        thickness_maps = {}
                        thicknesses = []
                        total_reasons = {}
                        for res in result:
                            th, th_map, reasons = res.get()
                            for method, reas in reasons.items():
                                if method not in total_reasons:
                                    total_reasons[method] = {}
                                for key, number in reas.items():
                                    if key not in total_reasons[method]:
                                        total_reasons[method][key] = 0
                                    total_reasons[method][key] += number

                            for method, thickness_map in th_map.items():
                                if method not in thickness_maps:
                                    thickness_maps[method] = thickness_map
                                else:
                                    thickness_maps[method] += thickness_map
                            thicknesses.extend(th)
            else:
                thicknesses, thickness_maps, reasons = estimate_thickness(tomo_data, tomo_seg, normals, points, cylinder_radius, cylinder_height, ps, 1000, 
                                                                        True, methods,profile_window_size,
                                                                            profile_sigma, check_output, 0, current_check_output_path,threshold_gradient_crop,
                                                                            maxima_gradient_crop, gradient_threshold)

            thicknesses = np.array(thicknesses).T
            for t, method in enumerate(out_methods):
                filtered_csv[method] = thicknesses[t]

            
            filtered_csv.set_index(['xyz_x_pixel', 'xyz_y_pixel', 'xyz_z_pixel'], inplace=True)
            csv.set_index(['xyz_x_pixel', 'xyz_y_pixel', 'xyz_z_pixel'], inplace=True)
            for method in out_methods:
                # if method not in csv:
                csv[method] = np.nan
                csv[method].update(filtered_csv[method])
            # Reset the index to bring back 'x', 'y', and 'z' as columns
            csv.reset_index(inplace=True)
                
            for t, method in enumerate(out_methods):
                orig_csv[method] = csv[method]


            orig_csv.to_csv(csv_file, sep=',', index=False)
        if use_ray:
            del tomo_data
        break

       

if __name__ == "__main__":
    pass
    # if len(argv) <= 1:
    #     print("First parameter has to be the config file.")
    # with open(argv[1]) as file:
    #     config = yaml.safe_load(file)
    #     if not config["data_dir"]:
    #         print("data_dir not specified in config.yml")
    #         exit()
    #     elif not config["data_dir"].endswith("/"):
    #         config["data_dir"] += "/"
    #     if not config["work_dir"]:
    #         print("work_dir not specified in config.yml - data_dir will be used for output")
    #         config["work_dir"] = config["data_dir"]
    #     elif not config["work_dir"].endswith("/"):
    #         config["work_dir"] += "/"
    # # See if a specific file was specified

    # if len(argv) == 2:
    #     print("No input file specified - will run on meshes for all segmentation files in the data directory")
    #     print("Pattern Matched: "+config["data_dir"]+"*.mrc")
    #     segmentation_files = glob.glob(config["data_dir"]+"*.mrc")
    #     segmentation_files = [os.path.basename(f) for f in segmentation_files]
    #     print(segmentation_files)
    #     tomograms = glob.glob(config["thickness_estimations"]["tomogram_dir"]+"*.mrc")


    # elif len(argv) == 3:
    #     print("Input file specified - will run on meshes associated with this segmentation only")
    #     segmentation_files = [argv[2]]
    #     tomograms = glob.glob(config["thickness_estimations"]["tomogram_dir"]+"*.mrc")
    # elif len(argv) == 4:
    #     segmentation_files = [argv[2]]
    #     tomograms = [argv[3]]

    # segmentation_files = sorted(segmentation_files)
    # tomograms = sorted(tomograms)

    # if len(tomograms) != len(segmentation_files):
    #     print("Number of segmentation and tomograms found not equal.")

    #     exit()


    # cylinder_radius = config["thickness_estimations"]["cylinder_radius"]
    # cylinder_height = config["thickness_estimations"]["cylinder_height"]
    # profile_window_size = config["thickness_estimations"]["profile_window_size"]
    # profile_sigma = config["thickness_estimations"]["profile_sigma"]
    # output_path = Path(config["work_dir"])
    # check_output_path = output_path / "check_output"
    # # check_output_path = config["thickness_estimations"]["check_output_path"]
    # check_output = config["thickness_estimations"]["check_output"]
    # threshold_gradient_crop = config["thickness_estimations"]["threshold_gradient_crop"]
    # maxima_gradient_crop = config["thickness_estimations"]["maxima_gradient_crop"]
    # high_pass = config["thickness_estimations"]["high_pass_filter"]
    # methods = config["thickness_estimations"]["methods"]
    # gradient_threshold = config["thickness_estimations"]["gradient_threshold"]
    # njobs = config["cores"]
    # use_ray = True
    # out_methods = [f"thickness_{method}" for method in methods]









    # if use_ray:
    #     tmp_dir =Path.home() / "ray"
    #     ray.init( _system_config={ 'automatic_object_spilling_enabled':False }, num_cpus=njobs, _temp_dir=str(tmp_dir))
    #     session = Path(tmp_dir) / "session_latest"
    #     session = session.resolve()

    # if check_output > 0:
    #     if not Path(check_output_path).exists():
    #         Path(check_output_path).mkdir(parents=True)


    # for seg_file, tomo_file in zip(segmentation_files, tomograms):

    #     tomo_file = Path(config["thickness_estimations"]["tomogram_dir"]) / tomo_file
    #     tomo_data_orig = mrcfile.open(tomo_file, permissive=True)
    #     if config["thickness_estimations"]["pixel_size"] == 0:
    #         ps = tomo_data_orig.voxel_size["x"]
    #     else:
    #         ps = config["thickness_estimations"]["pixel_size"]

    #     tomo_data_orig = tomo_data_orig.data  * 1


    #     if config["thickness_estimations"]["clip"]:
    #         mean = np.mean(tomo_data_orig)
    #         std = np.std(tomo_data_orig)
    #         clip_stds = config["thickness_estimations"]["clip_stds"]
    #         tomo_data = np.clip(tomo_data_orig, mean-std*clip_stds,mean+std*clip_stds)
    #     else:
    #         tomo_data = tomo_data_orig
    #     del tomo_data_orig

    #     tomo_data -= np.mean(tomo_data)
    #     tomo_data /= np.std(tomo_data)

    #     if high_pass > 0:
    #         sig = int(high_pass / ps)
    #         sig += (sig + 1) % 2
    #         tomo_data = gaussian_filter(tomo_data,0,ps) - gaussian_filter(tomo_data,sig,ps)

    #     if config["surface_generation"]["angstroms"]:
    #         conversion = ps 
    #     else:
    #         conversion = ps / 10

        


    #     csv_files = glob.glob(config["work_dir"] +  str(Path(seg_file).stem) +"*AVV_rh*.csv")
    #     csv_files = sorted(csv_files)
    #     # csv_files = [csv_files[2]]
    #     # csv_files = csv_files[:1]

    #     with mrcfile.open(config["data_dir"] + seg_file, permissive=True) as tomo_seg:
    #         tomo_seg = tomo_seg.data * 1
        


    #     if config["thickness_estimations"]["smooth"]:
    #         smooth_kernel = config["thickness_estimations"]["smooth_kernel"]
    #         gaus = matlab_style_gauss3D((smooth_kernel,smooth_kernel,smooth_kernel),smooth_kernel)

    #         tomo_data = signal.convolve(tomo_data, gaus, mode="same")
    #     if use_ray:
    #         ray.put(tomo_data)
    #         ray.put(tomo_seg)


    #     for csv_file in csv_files:
    #         # csv_file = config["work_dir"] + csv_file 

    #         name = Path(csv_file).name
    #         label = name.replace(str(Path(seg_file).stem), "")
    #         label = label.split(".AVV_")[0][1:]
            
    #         print(label)
    #         orig_csv = pd.read_csv(csv_file, sep=",")
    #         if all([method in orig_csv.columns for method in out_methods]) and not config["thickness_estimations"]["reestimate"]:
    #             continue
    #         csv = orig_csv.copy()
    #         csv["xyz_x_pixel"] = csv["xyz_x"] / conversion
    #         csv["xyz_y_pixel"] = csv["xyz_y"] / conversion
    #         csv["xyz_z_pixel"] = csv["xyz_z"] / conversion

    #         csv = csv.astype({"xyz_x_pixel":int,"xyz_y_pixel":int,"xyz_z_pixel":int})

    #         filtered_csv = csv.drop_duplicates(subset=["xyz_x_pixel","xyz_y_pixel","xyz_z_pixel"])


    #         x = np.array(filtered_csv["xyz_x_pixel"], dtype=int)
    #         y = np.array(filtered_csv["xyz_y_pixel"], dtype=int)
    #         z = np.array(filtered_csv["xyz_z_pixel"], dtype=int)

    #         normals = np.array((filtered_csv["normal_z"],filtered_csv["normal_y"],filtered_csv["normal_x"])).T
    #         points = np.array([z,y,x]).T

    #         current_check_output_path = Path(check_output_path) / Path(seg_file).stem / label
    #         if not current_check_output_path.exists():
    #             current_check_output_path.mkdir(parents=True)
    #         if njobs > 1:
    #             if use_ray:
    #                 points_per_njob = int(len(points) / njobs) + 1
    #                 points = [points[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)]
    #                 normals = [normals[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)] 

                    
    #                 result = [estimate_thickness.remote(tomo_data, tomo_seg, normal, point, cylinder_radius, cylinder_height, ps, 1000, True, 
    #                                                                     methods,profile_window_size, profile_sigma,
    #                                                                     check_output,process_id, current_check_output_path, threshold_gradient_crop, maxima_gradient_crop ) 
    #                                                                     for process_id, (point, normal) in enumerate(zip(points, normals))]
    #                 thickness_maps = {}
    #                 thicknesses = []
    #                 total_reasons = {}
    #                 for res in result:
    #                     th, th_map, reasons = ray.get(res)
    #                     for method, reas in reasons.items():
    #                         if method not in total_reasons:
    #                             total_reasons[method] = {}
    #                         for key, number in reas.items():
    #                             if key not in total_reasons[method]:
    #                                 total_reasons[method][key] = 0
    #                             total_reasons[method][key] += number

    #                     for method, thickness_map in th_map.items():
    #                         if method not in thickness_maps:
    #                             thickness_maps[method] = np.copy(thickness_map)
    #                         else:
    #                             thickness_maps[method] += np.copy(thickness_map)
    #                     thicknesses.extend(th)
    #             else:
    #                 with mp.Pool(njobs) as pool:
    #                     points_per_njob = int(len(points) / njobs) + 1
    #                     points = [points[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)]
    #                     normals = [normals[i*points_per_njob:(i+1)*points_per_njob] for i in range(njobs)] 

    #                     result = [pool.apply_async(estimate_thickness, args=[tomo_data, tomo_seg,  normal, point, cylinder_radius, cylinder_height, ps, 1000, True, 
    #                                                                         methods,profile_window_size, profile_sigma,
    #                                                                         check_output,process_id, current_check_output_path, threshold_gradient_crop, maxima_gradient_crop ]) 
    #                                                                         for process_id, (point, normal) in enumerate(zip(points, normals))]
    #                     thickness_maps = {}
    #                     thicknesses = []
    #                     total_reasons = {}
    #                     for res in result:
    #                         th, th_map, reasons = res.get()
    #                         for method, reas in reasons.items():
    #                             if method not in total_reasons:
    #                                 total_reasons[method] = {}
    #                             for key, number in reas.items():
    #                                 if key not in total_reasons[method]:
    #                                     total_reasons[method][key] = 0
    #                                 total_reasons[method][key] += number

    #                         for method, thickness_map in th_map.items():
    #                             if method not in thickness_maps:
    #                                 thickness_maps[method] = thickness_map
    #                             else:
    #                                 thickness_maps[method] += thickness_map
    #                         thicknesses.extend(th)
    #         else:
    #             thicknesses, thickness_maps, reasons = estimate_thickness(tomo_data, tomo_seg, normals, points, cylinder_radius, cylinder_height, ps, 1000, 
    #                                                                     True, methods,profile_window_size,
    #                                                                         profile_sigma, check_output, 0, current_check_output_path,threshold_gradient_crop,
    #                                                                         maxima_gradient_crop, gradient_threshold)

    #         thicknesses = np.array(thicknesses).T
    #         for t, method in enumerate(out_methods):
    #             filtered_csv[method] = thicknesses[t]
    #         # filtered_csv["thickness_global_median"] = thicknesses[0]
    #         # filtered_csv["thickness_local_median"]= thicknesses[1]
    #         # filtered_csv["thickness_maxima"]= thicknesses[2]
    #         # filtered_csv["thickness_local_mode"] = thicknesses[3]
            
    #         filtered_csv.set_index(['xyz_x_pixel', 'xyz_y_pixel', 'xyz_z_pixel'], inplace=True)
    #         csv.set_index(['xyz_x_pixel', 'xyz_y_pixel', 'xyz_z_pixel'], inplace=True)
    #         for method in out_methods:
    #             # if method not in csv:
    #             csv[method] = np.nan
    #             csv[method].update(filtered_csv[method])
    #         # Reset the index to bring back 'x', 'y', and 'z' as columns
    #         csv.reset_index(inplace=True)
                
    #         # csv = csv.merge(filtered_csv, on=['xyz_x_pixel', 'xyz_y_pixel', 'xyz_z_pixel'], how='left')
    #         for t, method in enumerate(out_methods):
    #             orig_csv[method] = csv[method]
    #         # orig_csv["thickness_global_median"] = csv["thickness_global_median"]
    #         # orig_csv["thickness_local_median"] = csv["thickness_local_median"]
    #         # orig_csv["thickness_maxima"] = csv["thickness_maxima"]
    #         # orig_csv["thickness_local_mode"] = csv["thickness_local_mode"]

    #         orig_csv.to_csv(csv_file, sep=',', index=False)
    #     if use_ray:
    #         del tomo_data
    #     break

    # if use_ray:
    #     ray.shutdown()
    #     if session.exists():
    #         shutil.rmtree(session, ignore_errors=True)