import numpy as np
import cv2
import bezier
from scipy.ndimage import distance_transform_edt
from torch import randint

from dataset.tamed_robot import TamedRobot
from dataset.mask_perturb import random_erode, get_random_structure


def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def get_boundary_scribble(region):
    # Draw along the boundary of an error region
    erode_size = np.random.randint(3, 50)
    eroded = cv2.erode(region, disk_kernel(erode_size))
    scribble = cv2.morphologyEx(eroded, cv2.MORPH_GRADIENT, np.ones((3,3)))

    h, w = region.shape
    for _ in range(4):
        lx, ly = np.random.randint(w), np.random.randint(h)
        lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)
        scribble[ly:lh, lx:lw] = random_erode(scribble[ly:lh, lx:lw], min=5)

    return scribble

def get_curve_scribble(region, min_srb=1, max_srb=4, sort=True, thickness=3):
    # Draw random curves
    num_lines = np.random.randint(min_srb, max_srb)

    scribbles = []
    lengths = []
    eval_pts = np.linspace(0.0, 1.0, 1024)
    if sort:
        # Generate more anyway, pick the best k at last
        num_gen = 10
    else:
        num_gen = num_lines
    region_indices = np.argwhere(region)
    if num_lines < 1:
        return np.zeros_like(region)
    for _ in range(num_gen):
        include_idx = np.random.choice(region_indices.shape[0], size=3, replace=False)
        y_nodes = np.asfortranarray([
            [0.0, 0.5, 1.0],
            region_indices[include_idx, 0],
        ])
        x_nodes = np.asfortranarray([
            [0.0, 0.5, 1.0],
            region_indices[include_idx, 1],
        ])
        x_curve = bezier.Curve(x_nodes, degree=2)
        y_curve = bezier.Curve(y_nodes, degree=2)
        x_pts = x_curve.evaluate_multi(eval_pts)
        y_pts = y_curve.evaluate_multi(eval_pts)

        this_scribble = np.zeros_like(region)
        pts = np.stack([x_pts[1,:], y_pts[1,:]], 1)
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        this_scribble = cv2.polylines(this_scribble, [pts], isClosed=False, color=(1), thickness=3)

        # Mask away path outside the allowed region, allow some error in labeling
        allowed_error = np.random.randint(3, 7)
        allowed_region = cv2.dilate(region, disk_kernel(allowed_error))
        this_scribble = this_scribble * allowed_region

        scribbles.append(this_scribble)
        lengths.append(this_scribble.sum())

    # Sort according to length, we want the long lines
    scribbles = [x for _, x in sorted(zip(lengths, scribbles), key=lambda pair: pair[0], reverse=True)]
    scribble = sum(scribbles[:num_lines])
    # print(type(scribble), num_gen, num_lines)
    return (scribble>0.5).astype(np.uint8)

def get_thinned_scribble(region):
    # Use the thinning algorithm for scribbles
    thinned = (cv2.ximgproc.thinning(region*255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)>128).astype(np.uint8)

    scribble = cv2.dilate(thinned, np.ones((3, 3)))
    h, w = region.shape
    for _ in range(4):
        lx, ly = np.random.randint(w), np.random.randint(h)
        lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)
        scribble[ly:lh, lx:lw] = random_erode(scribble[ly:lh, lx:lw], min=5)

    return scribble

def get_point_scribble(region, max_points=10, min_points=0, max_rad=8, min_rad=2):
    region_indices = np.argwhere(region)
    # rd = region.astype(int).sum()//20000
    # num_points = np.random.randint(min_points, min(max(rd*2, 2), max_points))
    num_points = np.random.randint(min_points, max_points)
    # print(region.sum(), num_points)
    srb = np.zeros_like(region)
    if num_points < 1:
        return srb
    include_idx = np.random.choice(region_indices.shape[0], size=num_points, replace=False)
    # size = np.random.randint(2, 10)
    for idx in include_idx:
        # print(idx)
        cv2.circle(srb, region_indices[idx][::-1], radius=np.random.randint(min_rad, max_rad), color=1, thickness=-1)
    return srb

def get_center_point(region, max_rad=4, min_rad=2, max_points=-1, min_points=-1):
    '''
    select 1 random point near the center
    note: min/max_points are placeholders since only 1 point is sampled here
    '''
    region_indices = np.argwhere(region)
    random_index = region_indices[np.random.randint(region_indices.shape[0])]
    # move toward a random point from the center
    select_index = ((np.mean(region_indices, axis=0) + random_index)*0.5).astype(int)
    if region[select_index[0], select_index[1]] == 0:
        # give up if the selected point is out of region
        select_index = random_index
        # print('random select: ', select_index)
    # srb = np.ascontiguousarray(region.astype(np.uint8))
    srb = np.zeros_like(region, dtype=np.uint8)
    cv2.circle(srb, select_index[::-1].tolist(), radius=np.random.randint(min_rad, max_rad), color=1, thickness=-1)
    return srb

def get_region_gt(gt, is_transition_included=False, tran_size_min=5, tran_size_max=40):
    gt_fg = gt >= 254
    gt_bg = gt <= 1
    if is_transition_included:
        size = np.random.randint(tran_size_min, tran_size_max)
        kernel = get_random_structure(size)
        # gt_tran = cv2.dilate((~(gt_fg|gt_bg)).astype(np.uint8), kernel) > 0
        gt_tran = (cv2.dilate(gt, kernel) - cv2.erode(gt, kernel)) > 0
        gt_fg &= ~gt_tran
        gt_bg &= ~gt_tran
        return [gt_fg, gt_bg, gt_tran]
    return [gt_fg, gt_bg]

def get_deform_regions(regions):
    size = np.random.randint(10, 50)
    kernel = get_random_structure(size)
    it = np.random.randint(2, 5)
    return [cv2.erode(reg.astype(np.uint8), kernel, iterations=it, borderValue=0) > 0 for reg in regions]

def get_region_revision(mask, gt_regions, from_zero=False, is_transition_included=False, is_bg_boundary=False):
    gt_fg, gt_bg= gt_regions[:2]

    # False positive and false negative
    if from_zero:
        fn = gt_fg.astype(np.uint8)
        fp = gt_bg.astype(np.uint8)
    else:
        # mask_fg, mask_bg, mask_tran = get_trimap(mask)
        threshold_jitter = np.random.randint(96, 127)
        threshold_high = 128+threshold_jitter
        threshold_low = 128-threshold_jitter
        mask_fg = mask >= threshold_high
        mask_bg = mask <= threshold_low
        fn = (~mask_fg & gt_fg).astype(np.uint8)
        fp = (~mask_bg & gt_bg).astype(np.uint8)

    if is_transition_included:
        gt_tran = gt_regions[-1]
        gt_fg &= ~gt_tran
        gt_bg &= ~gt_tran
        # tran = ~(gt_fg|gt_bg)&(gt_tran.astype(np.uint8))
        tran = gt_tran.astype(np.uint8)
        if is_bg_boundary:
            tran_dilated = cv2.dilate(tran, np.ones((17, 17)), iterations=np.random.randint(2, 6))
            fp &= tran_dilated
            gt_bg &= (tran_dilated > 0)
        return [fn, fp, tran]
    return [fn, fp]

def get_params(is_transition_included):
    fg_param = {
        'curve':
        {
            'thickness': 3,
            'min_srb': 1,
            'max_srb': 2,
        },
        'point':
        {
            'min_points': 1,
            'max_points': 3,
        }
    }

    bg_param = {
        'curve':
        {
            'thickness': 3,
            'min_srb': 1,
            'max_srb': 3,
        },
        'point':
        {
            'min_points': 1,
            'max_points': 3,
        }
    }
    
    params = [fg_param, bg_param]
    if is_transition_included:
        tran_param = {
            'curve':
            {
                'thickness': np.random.randint(5, 50),
                'min_srb': 1,
                'max_srb': 3,
            },
            'point':
            {
                'min_points': 1,
                'max_points': 5,
                'max_rad': 10,
            }
        }
        params.append(tran_param)
    
    return params

robot = TamedRobot()
def get_scribble(
    mask, gt, 
    from_zero=True, 
    is_transition_included=False, 
    is_connected_components=False,
    is_point_center_only=False,
    sample_regions=None,
):

    use_robot = False
    # if from_zero:
    #     use_robot = False
    # else:
    #     if np.random.rand() < 0.25:
    #         use_robot = True
    #     else:
    #         use_robot = False
    
    # Preprocessing
    gt_regions = get_region_gt(gt, is_transition_included=is_transition_included)

    if is_predef_regions:= (sample_regions is not None):
        assert len(gt_regions) == len(sample_regions)
    else:
        sample_regions = get_region_revision(
            mask, gt_regions, from_zero, is_transition_included, is_bg_boundary=np.random.rand() < 0.5)

    get_point_function = get_center_point if np.random.rand() < .5 or is_point_center_only else get_point_scribble

    # Generate scribbles
    if use_robot and not is_predef_regions:
        # The robot is similar to the DAVIS official one
        # pos_scr = robot.interact(fn).astype(np.uint8)
        # neg_scr = robot.interact(fp).astype(np.uint8)
        # pos_scr = cv2.dilate(pos_scr, np.ones((3,3)))
        # neg_scr = cv2.dilate(neg_scr, np.ones((3,3)))
        k = np.ones((3, 3))
        return [cv2.dilate(robot.interact(reg).astype(np.uint8), k) for reg in sample_regions]
    else:
        # Opening operator to remove noises
        if not is_predef_regions:
            opening_size = np.random.randint(5, 10)
            sample_regions[0] = cv2.morphologyEx(sample_regions[0], cv2.MORPH_OPEN, disk_kernel(opening_size))
            sample_regions[1] = cv2.morphologyEx(sample_regions[1], cv2.MORPH_OPEN, disk_kernel(opening_size))

        # Use connected error regions for processing
        params = get_params(is_transition_included)
        scribbles = []
        for m, param in zip(sample_regions, params): # FG, BG, Tran
            # Pack
            if is_connected_components:
                # Obtain scribble for this single region
                num_labels, labels_im = cv2.connectedComponents(m)
                regions = [(labels_im==i).astype(np.uint8) for i in range(1, num_labels)]
            else:
                # scribble for the full maksks
                regions = m if is_predef_regions else [m]
            
            # Get scribbles for each region
            this_scribble = np.zeros_like(gt)
            for region_mask in regions:
                if region_mask.sum() < np.random.randint(10, 300):
                    continue
                # Initial pass
                this_scribble |= get_a_random_scribble(region_mask, param, get_point_function)
                # Optionally use a second scribble type
                if np.random.rand() < 0.3:
                    this_scribble |= get_a_random_scribble(region_mask, param, get_point_function)

            scribbles.append(this_scribble)

        # Sometimes we just draw scribbles referring only to the GT but not the given mask
        if np.random.rand() < 0.5 or (scribbles[0].sum() == 0 and scribbles[1].sum() == 0):
            for i, m in enumerate(gt_regions):
                if m.sum() < 100:
                    continue
                scribbles[i] |= get_a_random_scribble(m.astype(np.uint8), params[i])

        # return scribbles[0].astype(np.uint8), scribbles[1].astype(np.uint8)
        return gt_regions, [srb.astype(np.uint8) for srb in scribbles]

def get_a_random_scribble(region_mask, param, get_point_function=get_point_scribble):
        # pick a scribble type
        pick = np.random.rand()
        # if pick < 0.15:
        #     region_scribble = get_boundary_scribble(region_mask)
        # elif pick < 0.2:
        #     region_scribble = get_thinned_scribble(region_mask)
        # elif pick < 0.3:
        if pick < 0.3:
            region_scribble = get_curve_scribble(region_mask, **param['curve'])
        else:
            region_scribble = get_point_function(region_mask, **param['point'])
        return region_scribble

if __name__ == '__main__':
    import sys
    mask = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    fp_scibble, fn_scibble = get_scribble(mask, gt, False)

    cv2.imwrite('s1.png', fp_scibble*255)
    cv2.imwrite('s2.png', fn_scibble*255)