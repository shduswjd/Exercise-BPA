import os
import glob
import cv2
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sklearn.svm
import scipy.ndimage
import itertools

def plot_multiple(images, titles=None, colormap='viridis', 
                  max_columns=np.inf, imwidth=4, imheight=4, share_axes=False):
    """Plot multiple images as subplots on a grid."""
    if titles is None:
        titles = [''] *len(images)
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * imwidth, n_rows * imheight),
        squeeze=False, sharex=share_axes, sharey=share_axes)

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis('off')
        
    if not isinstance(colormap, (list,tuple)):
        colormaps = [colormap]*n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    fig.tight_layout()

def load_dataset(dataset_dir):
    def natural_sort_key(s):
        return [float(t) if t.isdigit() else t for t in re.split('([0-9]+)', s)]
    
    def load_images(*path_parts):
        paths = glob.glob(os.path.join(dataset_dir, *path_parts))
        return [imageio.imread(p) for p in sorted(paths, key=natural_sort_key)]
        
    train_images_pos = load_images('TrainImages', 'pos-*.pgm')
    train_images_neg = load_images('TrainImages', 'neg-*.pgm')
    test_images = load_images('TestImages', 'test-*.pgm')
    assert (len(train_images_pos) == 550 and 
            len(train_images_neg) == 500 and
            len(test_images) == 170)
    return train_images_pos, train_images_neg, test_images

### CHANGE THIS TO THE DATASET PATH ###
dataset_dir = 'C:\Users\User\OneDrive\바탕 화면\python_workspace\ML\Exercise-BPA\CV\Exercise2\CarData'
train_images_pos, train_images_neg, test_images = load_dataset(dataset_dir)

# POINTS: 4

def image_gradients_polar(image):
    filter_kernel = np.array([[-1,0,1]], dtype=np.float32)
    dx = scipy.ndimage.convolve(image, filter_kernel, mode='reflect')
    dy = scipy.ndimage.convolve(image, filter_kernel.T, mode='reflect')
    magnitude = np.hypot(dx, dy)
    direction = np.arctan2(dy, dx) # between -pi and +pi
    return magnitude, direction

def hoglike_descriptor(image, cell_size=8, n_bins=16):
    image = image.astype(np.float32)/255
    grad_mag, grad_dir = image_gradients_polar(np.sqrt(image))

    # YOUR CODE HERE
    h, w= image.shape
    num_y = h // cell_size
    num_x = w // cell_size
    
    hog = np.zeros((num_y, num_x, n_bins), dtype=np.float32)
    
    for y in range(num_y):
        for x in range(num_x):
            cell_mag = grad_mag[y * cell_size:(y + 1) * cell_size,
                                x * cell_size:(x + 1) * cell_size]
            cell_dir = grad_dir[y * cell_size:(y + 1) * cell_size,
                                x * cell_size:(x + 1) * cell_size]

            hist, _ = np.histogram(cell_dir, bins=n_bins, range=(-np.pi, np.pi), weights=cell_mag)
            hog[y, x] = hist
    
    # Normalization
    bin_norm = np.linalg.norm(hog, axis=-1, keepdims=True)
    return hog / (bin_norm + 1e-4)

def draw_line(img, pt1, pt2, color, thickness=1):
    pt1 = tuple(np.round(pt1*16).astype(int))
    pt2 = tuple(np.round(pt2*16).astype(int))
    cv2.line(img, pt1, pt2, color=color, shift=4, 
             thickness=thickness, lineType=cv2.LINE_AA)

def plot_hog_cell(image_roi, hog_cell):
    """Visualize a single HOG cell."""
    output_size = image_roi.shape[0]
    half_bin_size = np.pi / len(hog_cell) / 2
    tangent_angles = np.linspace(0, np.pi, len(hog_cell), endpoint=False) + np.pi/2
    center = output_size / 2
    
    for cell_value, tangent_angle in zip(hog_cell, tangent_angles):
        cos_sin = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])
        offset = cell_value * output_size * cos_sin *0.5
        draw_line(image_roi, center - offset, center + offset, 
                  color=(249,129,42), thickness=3)

def plot_hog(image, hog, cell_size=8):
    upsample_factor = 96 / cell_size
    result = cv2.resize(image, (0, 0), fx=upsample_factor, fy=upsample_factor,
                        interpolation=cv2.INTER_NEAREST)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    result = (result.astype(np.float32)*0.6).astype(np.uint8)

    for y, x in np.ndindex(*hog.shape[:2]):
        yx = np.array([y, x])
        y0_out, x0_out = (yx * cell_size * upsample_factor).astype(int)
        y1_out, x1_out = ((yx+1) * cell_size * upsample_factor).astype(int)
        result_roi = result[y0_out:y1_out, x0_out:x1_out]
        plot_hog_cell(result_roi, hog[y, x])
    return result

# Two simple wave images are here to help understand the visualization
waves = [imageio.imread('sine.png'), imageio.imread('circular_sine.jpg')]
images = waves + train_images_pos[:6] + train_images_neg[:6]
hogs = [hoglike_descriptor(image) for image in images]
hog_plots = [plot_hog(image, hog) for image, hog in zip(images, hogs)]
titles = ['Wave 1', 'Wave 2'] + ['Positive']*6 + ['Negative']*6
plot_multiple(hog_plots, titles, max_columns=2, imheight=2, imwidth=4, share_axes=False)

def train_svm(positive_hog_windows, negative_hog_windows):
    svm = sklearn.svm.LinearSVC(C=0.01, loss='hinge', dual=True)
    hog_windows = np.concatenate([positive_hog_windows, negative_hog_windows])
    svm_input = hog_windows.reshape([len(hog_windows),-1])
    svm_target = np.concatenate((
            np.full(len(positive_hog_windows), 1, dtype=np.float32),
            np.full(len(negative_hog_windows), 0, dtype=np.float32)))
    svm.fit(svm_input, svm_target)
    return svm

def predict_svm(svm, hog_window):
    """Return the template response, i.e. the SVM's decision function without the sign."""
    return svm.decision_function(hog_window.reshape(1, -1))

start_time = time.time()
print('Computing features...')
positive_hog_windows = [hoglike_descriptor(im) for im in train_images_pos]
negative_hog_windows = [hoglike_descriptor(im) for im in train_images_neg]
duration = time.time()-start_time     
print(f'Done. Took {duration:.2f} s.')

start_time = time.time()
print('Training SVM...')
svm = train_svm(positive_hog_windows, negative_hog_windows)
duration = time.time()-start_time
print(f'Done. Took {duration:.2f} s.')

template = svm.coef_.reshape(positive_hog_windows[0].shape)  # reshape weight vector to shape of HOG-descriptor
template_pos = np.maximum(0, template) / template.max()      # we cannot visualize negative numbers, so separate
template_neg = np.maximum(0, -template) / -template.min()    # them for independent visualization
hog_plots = [
    plot_hog(np.zeros_like(train_images_pos[0]), template_pos),
    plot_hog(np.zeros_like(train_images_pos[0]), template_neg)
]
titles = ['positive weights', 'negative weights']
plot_multiple(hog_plots, titles=titles, max_columns=2, imheight=2, imwidth=4)

# POINTS: 3

def get_score_map(svm, hog, window_shape):
    score_map = np.zeros((hog.shape[0] - window_shape[0] + 1, hog.shape[1] - window_shape[1] + 1))

    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            hog_window = hog[y:y+window_shape[0], x:x+window_shape[1]].flatten()
            score_map[y, x] = predict_svm(svm, hog_window)
    return score_map

def score_map_to_detections(score_map, threshold):
    # YOUR CODE HERE
    score_map_np = np.array(score_map)
    ys, xs = np.where(score_map > threshold) #여기서 에러뜸
    scores = score_map_np[ys, xs]
    return ys, xs, scores

def draw_detections(image, ys, xs, scores, window_shape, cell_size=8):
    offset_size = 0
    
    h, w = image.shape[:2]
    scale_out = 5
    output_image = cv2.resize(
        image, (w*scale_out, h*scale_out), interpolation=cv2.INTER_NEAREST)
    if output_image.ndim < 3:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    output_image = (output_image.astype(np.float32)*0.6).astype(np.uint8)
    
    window_size_out = np.array(window_shape[::-1]) * cell_size * scale_out
    color = (197,255,0)
    
    for y, x, score in zip(ys, xs, scores):
        im_p0 = (np.array([x,y]) * cell_size + offset_size) * scale_out
        im_p1 = im_p0 + window_size_out
        cv2.rectangle(output_image, tuple(im_p0), tuple(im_p1),
                      color, thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(output_image, f'{score:.2f}', tuple(im_p0), 
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, color,
                    thickness=2, lineType=cv2.LINE_AA)
    return output_image

images, titles = [], []
window_shape = positive_hog_windows[0].shape[:2]

for test_image in test_images[25:40]:
    hog = hoglike_descriptor(test_image)
    score_map = get_score_map(svm, hog, window_shape)
    ys, xs, scores = score_map_to_detections(score_map, 0.4)
    detection_image = draw_detections(
        test_image, ys, xs, scores, window_shape)
    
    images += [plot_hog(test_image, hog), score_map, detection_image]
    titles += ['HOG', 'Score map', 'Detections']

plot_multiple(images, titles, max_columns=3, imheight=1.8, imwidth=3.2)

# POINTS: 3

def nms(score_map):
    min_score = score_map.min()
    h, w = score_map.shape
    detections = []

    for y in range(h):
        for x in range(w):
            score = score_map[y, x]

            # Ignore positions with minimum score
            if score == min_score:
                continue

            # Check if the current score is greater than or equal to its neighbors
            if score >= score_map[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)].max():
                detections.append((y, x, score))

    return detections

images, titles = [], []
for test_image in test_images[25:40]:
    hog = hoglike_descriptor(test_image)
    score_map = nms(get_score_map(svm, hog, window_shape))
    ys, xs, scores = score_map_to_detections(score_map, 0.4)
    detection_image = draw_detections(
        test_image, ys, xs, scores, window_shape)
    
    images += [plot_hog(test_image, hog), score_map, detection_image]
    titles += ['HOG', 'Score map after NMS', 'Detections after NMS']

plot_multiple(images, titles, max_columns=3, imheight=1.8, imwidth=3.2)

def evaluate(test_images, svm, window_shape, descriptor_func=hoglike_descriptor, 
             cell_size=8, threshold=0.4):
    # load and parse true locations from dataset
    with open('CarData/trueLocations.txt', 'r') as f:
        true_locations = f.read().splitlines()
    true_locations = [[[int(x) for x in point[1:-1].split(',')] for point in im_locs.split(': ')[1].split()] for im_locs in true_locations]
    assert len(true_locations) == len(test_images)
    obj_count = sum([len(im_locs) for im_locs in true_locations])
    pos_count = 0
    neg_count = 0
    # iterate over all images
    for i in range(len(true_locations)):
        # detect cars
        hog = descriptor_func(test_images[i])
        score_map = nms(get_score_map(svm, hog, window_shape))
        ys, xs, scores = score_map_to_detections(score_map, threshold)
        # compare detections to true locations
        for y, x in zip(ys, xs):
            correct = False
            for j, (y_gt, x_gt) in enumerate(true_locations[i]):
                x_diff = abs(x*cell_size - x_gt)
                y_diff = abs(y*cell_size - y_gt)
                x_axis = 0.25 * 100
                y_axis = 0.25 * 40
                if x_diff ** 2 / x_axis**2 + y_diff**2 / y_axis**2 <= 1:
                    correct = True
                    break
            if correct:
                pos_count += 1
                del true_locations[i][j]
            else:
                neg_count += 1
    recall = pos_count / obj_count
    precision = pos_count / (pos_count + neg_count)
    fmeasure = 2 * recall * precision / (recall + precision);

    print(f'Correct Detections: {pos_count}')
    print(f'Missing Detections: {obj_count - pos_count}')
    print(f'Wrong Detections: {neg_count}')
    print(f'Recall: {100*recall:.2f}%')
    print(f'Precision: {100*precision:.2f}%')
    print(f'F-Measure: {100*fmeasure:.2f}%')
        
evaluate(test_images, svm, window_shape, hoglike_descriptor, threshold=0.4)

# POINTS: 5

def hoglike_descriptor_with_interp(image, cell_size=8, n_bins=16):
    # YOUR CODE HERE
    image = image.astype(np.float32) / 255
    grad_mag, grad_dir = image_gradients_polar(np.sqrt(image))

    h, w = image.shape
    num_cells_y = h // cell_size
    num_cells_x = w // cell_size

    hog = np.zeros((num_cells_y, num_cells_x, n_bins), dtype=np.float32)

    for y in range(num_cells_y):
        for x in range(num_cells_x):
            y_start, y_end = y * cell_size, (y + 1) * cell_size
            x_start, x_end = x * cell_size, (x + 1) * cell_size

            # Compute histogram of oriented gradients for the current cell with bilinear interpolation
            for i in range(y_start, y_end):
                for j in range(x_start, x_end):
                    if i >= h or j >= w:
                        continue
                    cell_y_frac = (i - y_start + 0.5) / cell_size
                    cell_x_frac = (j - x_start + 0.5) / cell_size
                    cell_y_bin = int(cell_y_frac * n_bins)
                    cell_x_bin = int(cell_x_frac * n_bins)
                    bin_weight_y = cell_y_frac * n_bins - cell_y_bin
                    bin_weight_x = cell_x_frac * n_bins - cell_x_bin

                    for bin_y in range(2):
                        for bin_x in range(2):
                            cur_bin_y = min(max(cell_y_bin + bin_y, 0), n_bins - 1)
                            cur_bin_x = min(max(cell_x_bin + bin_x, 0), n_bins - 1)
                            weight = (1 - bin_weight_y if bin_y == 0 else bin_weight_y) * \
                                     (1 - bin_weight_x if bin_x == 0 else bin_weight_x)
                            hog[y, x, cur_bin_y % n_bins] += weight * grad_mag[i, j]
    
    # Normalization
    bin_norm = np.linalg.norm(hog, axis=-1, keepdims=True)
    return hog / (bin_norm + 1e-4)

start_time = time.time()
print('Computing features...')
descriptor_func = hoglike_descriptor_with_interp
positive_hog_windows = [descriptor_func(im) for im in train_images_pos]
negative_hog_windows = [descriptor_func(im) for im in train_images_neg]
duration = time.time()-start_time     
print(f'Done. Took {duration:.2f} s.')

start_time = time.time()
print('Training SVM...')
svm2 = train_svm(positive_hog_windows, negative_hog_windows)
duration = time.time()-start_time
print(f'Done. Took {duration:.2f} s.')

evaluate(test_images, svm2, window_shape, 
         hoglike_descriptor_with_interp, threshold=0.4)


def to_gray(im):
    if im.ndim < 3:
        return im
    else:
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

streetview_images = [imageio.imread(f'streetview{i}.jpg') for i in range(1,4)]
some_uiuc_test_images = test_images[:33]

input_images = streetview_images + some_uiuc_test_images
detection_images = []

for test_image in input_images:
    hog = hoglike_descriptor_with_interp(to_gray(test_image))
    score_map = nms(get_score_map(svm2, hog, window_shape))
    xs, ys, scores = score_map_to_detections(score_map, 0.8)
    detection_image = draw_detections(
        test_image, xs, ys, scores, window_shape)
    detection_images.append(detection_image)

plot_multiple(detection_images[:3], max_columns=1, imheight=2, imwidth=6)
plot_multiple(detection_images[3:], max_columns=3, imheight=1.8, imwidth=3.2)