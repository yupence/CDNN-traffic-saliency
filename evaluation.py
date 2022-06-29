'''
Created on 1 mar 2017

@author: 	Dario Zanca 
@summary: 	Collection of functions to compute visual attention metrics for:
                - saliency maps similarity
                    - AUC Judd (Area Under the ROC Curve, Judd version)
                    - KL Kullback Leiber divergence
                    - NSS Normalized Scanpath Similarity

                    update:
                    - Similiarity metric
                    - Correlation Coefficienct

                - scanpaths similarity
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import sys
import pandas as pd

##############################  saliency metrics  #######################################


''' created: Tilke Judd, Oct 2009
    updated: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017

This measures how well the saliencyMap of an image predicts the ground truth human 
fixations on the image. ROC curve created by sweeping through threshold values determined
by range of saliency map values at fixation locations;
true positive (tp) rate correspond to the ratio of saliency map values above threshold 
at fixation locations to the total number of fixation locations, false positive (fp) rate
correspond to the ratio of saliency map values above threshold at all other locations to 
the total number of posible other locations (non-fixated image pixels) '''


def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap
    new_size = np.shape(fixationMap)
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        # from scipy.misc import imresize
        new_size = np.shape(fixationMap)
        np.array(Image.fromarray(saliencyMap).resize(
            (new_size[1], new_size[0])))

        # saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + \
            np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
        / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    # sort sal map values, to sweep through values
    allthreshes = sorted(Sth, reverse=True)
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        # total number of sal map values above threshold
        aboveth = (S >= thresh).sum()
        # ratio sal map values at fixation locations
        tp[i + 1] = float(i + 1) / Nfixations
        # above threshold
        # ratio other sal map values
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score


######################################################################################

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017

This finds the KL-divergence between two different saliency maps when viewed as 
distributions: it is a non-symmetric measure of the information lost when saliencyMap 
is used to estimate fixationMap. '''


def KLdiv(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map

    # convert to float
    map1 = saliencyMap.astype(float)
    map2 = fixationMap.astype(float)

    # make sure maps have the same shape
    # from scipy.misc import imresize

    new_size = np.shape(map2)
    np.array(Image.fromarray(map1).resize((new_size[1], new_size[0])))

    # map1 = imresize(map1, np.shape(map2))

    # make sure map1 and map2 sum to 1
    if map1.any():
        map1 = map1 / map1.sum()
    if map2.any():
        map2 = map2 / map2.sum()

    # compute KL-divergence
    eps = 10 ** -12
    score = map2 * np.log(eps + map2 / (map1 + eps))

    return score.sum()


######################################################################################

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017

This finds the normalized scanpath saliency (NSS) between two different saliency maps. 
NSS is the average of the response values at human eye positions in a model saliency 
map that has been normalized to have zero mean and unit standard deviation. '''


def NSS(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = np.nan
        return score

    # make sure maps have the same shape
    # from scipy.misc import imresize

    new_size = np.shape(fixationMap)
    map1 = np.array(Image.fromarray(
        saliencyMap).resize((new_size[1], new_size[0])))

    # map1 = imresize(saliencyMap, np.shape(fixationMap))
    if not map1.max() == 0:
        map1 = map1.astype(float) / map1.max()

    # normalize saliency map
    if not map1.std(ddof=1) == 0:
        map1 = (map1 - map1.mean()) / map1.std(ddof=1)

    # mean value at fixation locations
    score = map1[fixationMap.astype(bool)].mean()

    return score


def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output


def NSS_gt(smap, gt):
    # 使用NSS方法评估显著性算法
    # 注意:
    # 1.输入smap必须为连续值,输入gt必须为特征图
    std = np.std(smap)
    u = np.mean(smap)
    smap = (smap - u) / std
    gt = image_binary(gt, 0.1)
    nss = smap * gt
    nss = np.sum(nss) / np.sum(gt)
    return nss


######################## created by salgan - added by memoona ##############################

def CC(s_map, gt):
    # gt is an empirical/continuous saliency map
    # make sure maps have the same shape
    s_map = s_map.astype(float)
    gt = gt.astype(float)

    # from scipy.misc import imresize

    new_size = np.shape(gt)
    np.array(Image.fromarray(s_map).resize((new_size[1], new_size[0])))

    # s_map = imresize(s_map, np.shape(gt))
    gt_norm = (gt - np.mean(gt)) / np.std(gt)

    # because some saliency maps had no fixation so it was empty and didnt require normalization
    if not s_map.max() == 0:
        s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
        r = (s_map_norm * gt_norm).sum() / math.sqrt((s_map_norm *
                                                      s_map_norm).sum() * (gt_norm * gt_norm).sum())
    else:
        r = 0

    # r = np.corrcoef(s_map_norm, gt_norm)
    return r


# created by salgan- added by memoona

def SIM(s_map, gt):
    # here gt is not discretized nor normalized
    s_map = s_map.astype(float)
    gt = gt.astype(float)

    # from scipy.misc import imresize

    new_size = np.shape(gt)
    np.array(Image.fromarray(s_map).resize((new_size[1], new_size[0])))

    # s_map = imresize(s_map, np.shape(gt))

    gt = (gt - np.min(gt)) / ((np.max(gt) - np.min(gt)) * 1.0)
    if not s_map.max() == 0:
        s_map = (s_map - np.min(s_map)) / \
            ((np.max(s_map) - np.min(s_map)) * 1.0)
        s_map = s_map / (np.sum(s_map) * 1.0)
        gt = gt / (np.sum(gt) * 1.0)
        x, y = np.where(gt > 0)
        sim = 0.0
        for i in zip(x, y):
            sim = sim + min(gt[i[0], i[1]], s_map[i[0], i[1]])
    else:
        sim = 0
    return sim


#########################################################################################

##############################  scanpaths metrics  ######################################

#########################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the Euclidean distance between two scanpath of the same length. '''


def euclidean_distance(human_scanpath, simulated_scanpath):
    if len(human_scanpath) == len(simulated_scanpath):

        dist = np.zeros(len(human_scanpath))
        for i in range(len(human_scanpath)):
            P = human_scanpath[i]
            Q = simulated_scanpath[i]
            dist[i] = np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
        return dist

    else:

        print('Error: The two sequences must have the same length!')
        return False


#########################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the string edit distance metric.

    Given an image, it is divided in nxn regions. To each region, a letter is assigned. 
    For each scanpath, the correspondent letter is assigned to each fixation, depending 
    the region in which such fixation falls. So that each scanpath is associated to a 
    string. 

    Distance between the two generated string is then compared as described in 
    "speech and language processing", Jurafsky, Martin. Cap. 3, par. 11. '''


def _Levenshtein_Dmatrix_initializer(len1, len2):
    Dmatrix = []

    for i in range(len1):
        Dmatrix.append([0] * len2)

    for i in range(len1):
        Dmatrix[i][0] = i

    for j in range(len2):
        Dmatrix[0][j] = j

    return Dmatrix


def _Levenshtein_cost_step(Dmatrix, string_1, string_2, i, j, substitution_cost=1):
    char_1 = string_1[i - 1]
    char_2 = string_2[j - 1]

    # insertion
    insertion = Dmatrix[i - 1][j] + 1
    # deletion
    deletion = Dmatrix[i][j - 1] + 1
    # substitution
    substitution = Dmatrix[i - 1][j - 1] + \
        substitution_cost * (char_1 != char_2)

    # pick the cheapest
    Dmatrix[i][j] = min(insertion, deletion, substitution)


def _Levenshtein(string_1, string_2, substitution_cost=1):
    # get strings lengths and initialize Distances-matrix
    len1 = len(string_1)
    len2 = len(string_2)
    Dmatrix = _Levenshtein_Dmatrix_initializer(len1 + 1, len2 + 1)

    # compute cost for each step in dynamic programming
    for i in range(len1):
        for j in range(len2):
            _Levenshtein_cost_step(Dmatrix,
                                   string_1, string_2,
                                   i + 1, j + 1,
                                   substitution_cost=substitution_cost)

    if substitution_cost == 1:
        max_dist = max(len1, len2)
    elif substitution_cost == 2:
        max_dist = len1 + len2

    return Dmatrix[len1][len2]


def _scanpath_to_string(scanpath, height, width, n):
    height_step, width_step = height // n, width // n

    string = ''

    for i in range(np.shape(scanpath)[0]):
        fixation = scanpath[i].astype(np.int32)
        correspondent_square = (
            fixation[0] / width_step) + (fixation[1] / height_step) * n
        string += chr(97 + correspondent_square)

    return string


def string_edit_distance(stimulus,  # matrix

                         human_scanpath, simulated_scanpath,

                         n=5,  # divide stimulus in a nxn grid
                         substitution_cost=1
                         ):
    height, width = np.shape(stimulus)[0:2]

    string_1 = _scanpath_to_string(human_scanpath, height, width, n)
    string_2 = _scanpath_to_string(simulated_scanpath, height, width, n)

    print(string_1, string_2)

    return _Levenshtein(string_1, string_2)


#########################################################################################

''' created: Dario Zanca, July 2017

    Implementation of the metric described in "Simulating Human Saccadic 
    Scanpaths on Natural Images", by Wei Wang, Cheng Chen, Yizhou Wang, 
    Tingting Jiang, Fang Fang, Yuan Yao 

    Time-delay embedding are used in order to quantitatively compare the 
    stochastic and dynamic scanpaths of varied lengths '''


def time_delay_embedding_distance(
        human_scanpath,
        simulated_scanpath,

        # options
        k=3,  # time-embedding vector dimension
        distance_mode='Mean'
):
    # human_scanpath and simulated_scanpath can have different lenghts
    # They are list of fixations, that is couple of coordinates
    # k must be shorter than both lists lenghts

    # we check for k be smaller or equal then the lenghts of the two input scanpaths
    if len(human_scanpath) < k or len(simulated_scanpath) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths

    human_scanpath_vectors = []
    for i in np.arange(0, len(human_scanpath) - k + 1):
        human_scanpath_vectors.append(human_scanpath[i:i + k])

    simulated_scanpath_vectors = []
    for i in np.arange(0, len(simulated_scanpath) - k + 1):
        simulated_scanpath_vectors.append(simulated_scanpath[i:i + k])

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k

    distances = []

    for s_k_vec in simulated_scanpath_vectors:

        # find human k-vec of minimum distance

        norms = []

        for h_k_vec in human_scanpath_vectors:
            d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
            norms.append(d)

        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False


def scaled_time_delay_embedding_distance(
        human_scanpath,
        simulated_scanpath,
        image,

        # options
        toPlot=False):
    # to preserve data, we work on copies of the lists
    H_scanpath = copy(human_scanpath)
    S_scanpath = copy(simulated_scanpath)

    # First, coordinates are rescaled as to an image with maximum dimension 1
    # This is because, clearly, smaller images would produce smaller distances
    max_dim = float(max(np.shape(image)))

    for P in H_scanpath:
        P[0] /= max_dim
        P[1] /= max_dim

    for P in S_scanpath:
        P[0] /= max_dim
        P[1] /= max_dim

    # Then, scanpath similarity is computer for all possible k
    max_k = min(len(H_scanpath), len(S_scanpath))
    similarities = []
    for k in np.arange(1, max_k + 1):
        s = time_delay_embedding_distance(
            H_scanpath,
            S_scanpath,
            k=k,  # time-embedding vector dimension
            distance_mode='Mean')
        similarities.append(np.exp(-s))
        print(similarities[-1])

    # Now that we have similarity measure for all possible k
    # we compute and return the mean

    if toPlot:
        keys = np.arange(1, max_k + 1)
        plt.plot(keys, similarities)
        plt.show()

    return sum(similarities) / len(similarities)


def KL_divergence(X, Y, nbins):
    """
    Calculate Kullback-Leibler divergence of X and Y
    Formula: KLD(X, Y) = D_KL (Px || Py) = sum(px * log2 (px / py))
    """
    amax = max(np.max(X), np.max(Y))
    amin = min(np.min(X), np.min(Y))
    p, _ = np.histogram(X, bins=nbins, range=[amin, amax])
    q, _ = np.histogram(Y, bins=nbins, range=[amin, amax])
    p = p / p.sum()
    q = q / q.sum()

    kld = 0

    for i in range(len(p)):
        if (p[i] > 0) and (q[i] > 0):
            kld += p[i] * np.log2(p[i] / q[i])
    return kld


def normalize_range(map):
    '''
    normalize a density map so each element be in the range of 0 and 1 such that the maximum value in matrix be 1
    :param map:
    :return:
    '''
    return (map - map.min()) / (map.max() - map.min() + 1e-5)


def AUC_Borji(saliency, fix_map, n_split=100, step_size=.1, ):
    '''
        compute the auc borji score
    :param saliency:
    :param fix_map:
    :param n_split:
    :param step_size:
    :return:
    '''

    if saliency.shape != fix_map.shape:
        saliency = scipy.misc.imresize(saliency, fix_map.shape)
    fix_map = fix_map.flatten().astype(np.bool)

    saliency = normalize_range(saliency.flatten().astype(np.float))
    sal_on_fix = np.sort(saliency[fix_map])

    r = np.random.randint(0, saliency.shape[0], (sal_on_fix.shape[0], n_split))
    rand_sal = saliency[r]
    auc = np.zeros(n_split)

    for i in range(n_split):
        cur_sal = rand_sal[:, i]
        sorted_cur_fix = np.sort(cur_sal)

        max_val = np.maximum(cur_sal.max(), sal_on_fix.max())
        tmp_treshold = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros(tmp_treshold.shape[0])
        fp = np.zeros(tmp_treshold.shape[0])

        for ind, ths in enumerate(tmp_treshold):
            tp[ind] = (sal_on_fix.shape[0] - sal_on_fix.searchsorted(ths,
                                                                     side='left')) * 1. / sal_on_fix.shape[0]
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(ths, side='left')) * 1. / sal_on_fix.shape[
                0]
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)


def IG(saliency_map, fixation_map, baseline_map=np.zeros((192, 320))):
    '''
        (192,320)
    :param saliency_map:
    :param fixation_map:
    :param baseline_map:
    :return:
'''

    if not isinstance(saliency_map, np.ndarray):
        saliency_map = np.array(saliency_map, dtype=np.float32)
    elif saliency_map.dtype != np.float32:
        saliency_map = saliency_map.astype(np.float32)

    if not isinstance(fixation_map, np.ndarray):
        fixation_map = np.array(fixation_map, dtype=np.float32)
    elif fixation_map.dtype != np.float32:
        fixation_map = fixation_map.astype(np.float32)

    if not isinstance(baseline_map, np.ndarray):
        baseline_map = np.array(baseline_map, dtype=np.float32)
    elif fixation_map.dtype != np.float32:
        baseline_map = baseline_map.astype(np.float32)

    saliency_map = (saliency_map - saliency_map.min()) / \
        (saliency_map.max() - saliency_map.min())

    saliency_map = saliency_map / saliency_map.sum()
    #
    # baseline_map = (baseline_map - baseline_map.min()) / (baseline_map.max() - baseline_map.min())
    # baseline_map = baseline_map / baseline_map.sum()

    fixs = fixation_map.astype(np.bool)

    EPS = np.finfo(np.float32).eps

    return (np.log2(EPS + saliency_map[fixs]) - np.log2(EPS + baseline_map[fixs])).mean()


def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""

    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i, j], i, j])
            count += 1
    return sig


def process_bar(num, total):
    rate = float(num) / total
    ratenum = int(100 * rate)
    r = '\r[{}{}]{}%'.format('*' * ratenum, ' ' * (100 - ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()


if __name__ == "__main__":
    data = np.load(r"./ckpts/cdnn/test.npz")
    outputs = data['p']
    targets = data['t']
    df = pd.DataFrame(columns=['AUC_Judd', 'AUC_Borji',
                               'NSS', 'IG', 'CC', 'SIM', 'KLD'], dtype=float)
    count = 0
    n = len(outputs)
    for output, target in zip(outputs, targets):
        Judd_item = AUC_Judd(output, target)
        Borji_item = AUC_Borji(output, target)
        CC_item = CC(output, target)
        NSS_item = NSS_gt(output, target)
        IG_item = IG(output, target)
        SIM_item = SIM(output, target)
        KLD_item = KLdiv(output, target)
        item = [Judd_item, Borji_item, NSS_item,
                IG_item, CC_item, SIM_item, KLD_item]
        df.loc[count] = item
        process_bar(count, n)
        count = count+1
    df.to_csv(r'./ckpts/evaluation.csv')
