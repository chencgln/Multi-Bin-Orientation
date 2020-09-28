import math as m
import numpy as np
from config.config import cfg

BINS = cfg.MODEL.ORIEN_BINS

def read_class_names_map():
    class_file_name = cfg.DATASET.CLASSES
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_dataset_list(list_file):
    image_list = []; label_list = []
    with open(list_file, "r") as ff:
        for sample in ff.readlines():
            image_file, label_file = sample.split()[:2]
            with open(label_file, 'r') as lf:
                for line in lf.readlines():
                    line_split = line.split()
                    if line_split[0] not in read_class_names_map().values():
                        continue
                    line_split[1:] = [float(ii) for ii in line_split[1:]]
                    bbox, alpha, dims, ry = line_split[4:8], line_split[3], line_split[8:11], line_split[14]
                    orien_label = make_orien_label(alpha)
                    image_list.append(image_file)
                    label_list.append(bbox+orien_label)
    return image_list, label_list

def make_orien_label(alpha):
    '''
    return orientation label as list, [confs]*BINS+[sin_off, cos_off]*BINS
    '''
    BINS = cfg.MODEL.ORIEN_BINS
    OVERLAP = cfg.MODEL.ORIEN_BIN_OVERLAP
    
    def compute_anchors(rad):
        '''
        Find bin which the rad falls in. Return bins' index and the offset to the bin;
        Length of each bin: [-wedge/2*(1+OVERLAP/2), wedge/2*(1+OVERLAP/2)]
        '''
        anchors = []
        wedge = 2.*np.pi/BINS
        l_index = int(rad/wedge)
        r_index = l_index + 1

        if (rad - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
            anchors.append([l_index, rad - l_index*wedge])
            
        if (r_index*wedge - rad) < wedge/2 * (1+OVERLAP/2):
            anchors.append([r_index%BINS, rad - r_index*wedge])
            
        return anchors

    def rect_orien(rad):
        '''
        Convert rad from [-pi,pi] to [0,2*pi]
        '''
        rad = rad + np.pi/2.
        if rad < 0:
            rad = rad + 2.*np.pi
        rad = rad - int(rad/(2.*np.pi))*(2.*np.pi)
        return rad

    orientation = np.zeros(BINS*2)
    confidence = np.zeros(BINS)

    alpha = rect_orien(alpha)
    anchors = compute_anchors(alpha)

    for anchor in anchors:
        orientation[[anchor[0]*2, anchor[0]*2+1]] = np.array([np.sin(anchor[1]), np.cos(anchor[1])])
        confidence[anchor[0]] = 1.
    confidence = confidence / np.sum(confidence)
    
    return list(confidence)+list(orientation)

def parse_multi_bin(confs, offsets):
    '''
    param
        confs: confidence of each bin (BINS)
        offsets: sin & cos value of offset to each bin (BINS*2)
    '''
    assert(confs.shape==(BINS,) and (offsets.shape==(BINS*2,) or offsets.shape==(BINS, 2)))
    offsets = offsets.reshape(BINS, 2)
    
    anchor = np.argmax(confs)
    offset_sin_cos = offsets[anchor]
    offset = m.atan2(offset_sin_cos[0], offset_sin_cos[1])
    wedge = 2.*np.pi/BINS
    alpha = offset + anchor*wedge

    ## convert back to Kitti's [-pi, pi]
    alpha = alpha % (2.*np.pi)
    alpha -= np.pi/2
    if alpha > np.pi:
        alpha = alpha - (2.*np.pi)

    return alpha