# Some imports and helper functions
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import cv2
import gco

def draw_mask_on_image(image, mask, color=(0, 255, 255)):
    """Return a visualization of a mask overlaid on an image."""
    result = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, kernel)
    outline = dilated > mask
    result[mask == 1] = (result[mask == 1] * 0.4 + 
                         np.array(color) * 0.6).astype(np.uint8)
    result[outline] = color
    return result

# POINTS: 1

im = imageio.imread('lotus320.jpg')
h,w = im.shape[:2]

# Set up initial foreground and background
# regions for building the color model
init_fg_mask = np.zeros([h, w]) #(320, 427)
init_bg_mask = np.zeros([h, w])


# Now set some rectangular region of the initial foreground mask to 1.
# This should be a part of the image that is fully foreground.
# The indices in the following line are just an example,
# and they need to be corrected so that only flower pixels are included
init_fg_mask[100:150, 150:200] = 1

# Same for the background (replace the indices)
init_bg_mask[60:90, 50:90] = 1


fig, axes = plt.subplots(1, 2, figsize=(9.5,5))
axes[0].set_title('Initial foreground mask')
axes[0].imshow(draw_mask_on_image(im, init_fg_mask))
axes[1].set_title('Initial background mask')
axes[1].imshow(draw_mask_on_image(im, init_bg_mask))
fig.tight_layout()


def calculate_histogram(im, mask, n_bins):
    histogram = np.full((n_bins, n_bins, n_bins), fill_value=0.001)
    # im : input image values in the range[0, 255]
    # mask: same size as the image
    # n_bins: bins used in the histogram along each dimension 
    # return : [n_bins, n_bins, n_bins]
    
    # normalize the image to the range[0, n_bins-1]
    norm_im = (im/256.0) * n_bins
    norm_im = norm_im.astype(int)
    
    # iterate over each pixel in the image
    for r in range(im.shape[0]):
        for g in range(im.shape[1]):
            if mask[r, g]: #check if the pixel is within the mask
                red, green, blue = norm_im[r, g]
                histogram[red, green, blue] += 1
    
    return histogram

n_bins = 10
fg_histogram = calculate_histogram(im, init_fg_mask, n_bins)
bg_histogram = calculate_histogram(im, init_bg_mask, n_bins)

fig, axes = plt.subplots(
    3, 2, figsize=(5,5), sharex=True, 
    sharey=True, num='Relative frequency of color bins')

x = np.arange(n_bins)
axes[0,0].bar(x, np.sum(fg_histogram, (1, 2)))
axes[0,0].set_title('red (foreground)')
axes[1,0].bar(x, np.sum(fg_histogram, (0, 2)))
axes[1,0].set_title('green (foreground)')
axes[2,0].bar(x, np.sum(fg_histogram, (0, 1)))
axes[2,0].set_title('blue (foreground)')

axes[0,1].bar(x, np.sum(bg_histogram, (1, 2)))
axes[0,1].set_title('red (background)')
axes[1,1].bar(x, np.sum(bg_histogram, (0, 2)))
axes[1,1].set_title('green (background)')
axes[2,1].bar(x, np.sum(bg_histogram, (0, 1)))
axes[2,1].set_title('blue (background)')
fig.tight_layout()

# POINTS: 4

def foreground_pmap(im, fg_histogram, bg_histogram):
    # YOUR CODE HERE
    n_bins = fg_histogram.shape[0]
    
    # normalize the image to the range[0, n_bins-1]
    norm_im = (im/256.0) * n_bins
    norm_im = norm_im.astype(int)
    
    # initialize the probability map
    probability_map = np.zeros((im.shape[0], im.shape[1]))
    
    # iterate over each pixel in the image
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            red, green, blue = norm_im[i, j]
            
            # retrieve the histogram values for the current pixel
            fg_value = fg_histogram[red, green, blue]
            bg_value = bg_histogram[red, green, blue]
            
            # calculate probability of pixel belong to the foreground
            probability = fg_value / (fg_value + bg_value)
            
            # 확률 저장
            probability_map[i, j] = probability
    return probability_map
    
foreground_prob = foreground_pmap(im, fg_histogram, bg_histogram)
# fig, axes = plt.subplots(1, 2, figsize=(9.5,5), sharey=True)
# axes[0].imshow(im)
# axes[0].set_title('Input image')
# im_plot = axes[1].imshow(foreground_prob, cmap='viridis')
# axes[1].set_title('Foreground posterior probability')
# fig.tight_layout()
# fig.colorbar(im_plot, ax=axes)
# foreground_map = (foreground_prob > 0.5)


def unary_potentials(probability_map, unary_weight):
    # YOUR CODE HERE
    return unary_weight * (-np.log(probability_map))
    
    
unary_weight = 1
unary_fg = unary_potentials(foreground_prob, unary_weight)
unary_bg = unary_potentials(1 - foreground_prob, unary_weight)
# fig, axes = plt.subplots(1, 2, figsize=(9.5,5), sharey=True)
# axes[0].imshow(unary_fg)
# axes[0].set_title('Unary potentials (foreground)')
# im_plot = axes[1].imshow(unary_bg)
# axes[1].set_title('Unary potentials (background)')
# fig.tight_layout()
# fig.colorbar(im_plot, ax=axes)


def pairwise_potential_prefactor(im, x1, y1, x2, y2, pairwise_weight):
    # YOUR CODE HERE
    # 두 픽셀의 색상 값 추출
    pixel1 = im[y1, x1]
    pixel2 = im[y2, x2]
    
    # 두 픽셀의 색상 값 차이 계산 
    diff = pixel1 - pixel2
    
    # 색상 값 차이의 제곱 합 계산 (거리 계산)
    diff_squared = np.sum(diff ** 2)
    
    # pairwise potential prefactor 계산
    w_d = 0.1
    return pairwise_weight * np.exp(-w_d * diff_squared)



def coords_to_index(x, y, width):
    return y * width + x

def pairwise_potentials(im, pairwise_weight):
    # YOUR CODE HERE
    height, width, _ = im.shape
    edges = []
    costs = []
    
    for y in range(height):
        for x in range(width):
            index1 = coords_to_index(x, y, width)
            pixel1 = im[y, x]
            
            if x + 1 < width:
                index2 = coords_to_index(x+1, y, width)
                pixel2 = im[y, x+1]
                cost = pairwise_potential_prefactor(im, x, y, x+1, y, pairwise_weight)
                edges.append([index1, index2])
                costs.append(cost)
            if y +  1 < height:
                index2 = coords_to_index(x, y+1, width)
                pixel2 = im[y+1, x]
                cost = pairwise_potential_prefactor(im, x, y, x, y+1, pairwise_weight)
                edges.append([index1, index2])
                costs.append(cost)
    edges = np.array(edges, dtype=int)
    costs = np.array(costs)
    
    return edges, costs

pairwise_edges, pairwise_costs = pairwise_potentials(im, pairwise_weight=1)

def graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs):
    unaries = np.stack([unary_bg.flat, unary_fg.flat], axis=-1)
    labels = gco.cut_general_graph(
        pairwise_edges, pairwise_costs, unaries, 
        1-np.eye(2), n_iter=-1, algorithm='swap')
    return labels.reshape(unary_fg.shape)

graph_cut_result = graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs)
# fig, axes = plt.subplots(1, 2, figsize=(9.5,5), sharey=True)
# axes[0].set_title('Thresholding of per-pixel foreground probability at 0.5')
# axes[0].imshow(draw_mask_on_image(im, foreground_prob>0.5))
# axes[1].set_title('Graph cut result')
# axes[1].imshow(draw_mask_on_image(im, graph_cut_result))
# fig.tight_layout()


def segment_image(im, init_fg_mask, init_bg_mask,
                  unary_weight, pairwise_weight, n_bins):
    # YOUR CODE HERE
    fg_histogram = calculate_histogram(im, init_fg_mask, n_bins)
    bg_histogram = calculate_histogram(im, init_bg_mask, n_bins)
    
    foreground_prob = foreground_pmap(im, fg_histogram, bg_histogram) 
    
    unary_fg = unary_potentials(foreground_prob, unary_weight)
    unary_bg = unary_potentials(1 - foreground_prob, unary_weight)
    # print("Unary potentials (foreground):", unary_fg)
    # print("Unary potentials (background):", unary_bg)
    
    height, width, _ = im.shape
    pairwise_edges, pairwise_costs = pairwise_potentials(im, pairwise_weight)
    segmented_mask =  graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs)
    # print("Graph cut result:", segmented_mask)
    
    return segmented_mask


import skimage.data

im_cells = skimage.data.immunohistochemistry()
h_cells, w_cells = im_cells.shape[:2]
fg_mask_cells = np.zeros([h_cells, w_cells])
bg_mask_cells = np.zeros([h_cells, w_cells])

# Set some appropriate parts of fg_mask and bg_mask to 1 for initialization.
# YOUR CODE HERE

def initialize_masks(h_cells, w_cells):
    fg_mask_cells = np.zeros([h_cells, w_cells], dtype=np.uint8)
    bg_mask_cells = np.zeros([h_cells, w_cells], dtype=np.uint8)

    # Define regions for foreground and background masks
    fg_start_row, fg_end_row = h_cells // 4, 3 * h_cells // 4
    fg_start_col, fg_end_col = w_cells // 4, 3 * w_cells // 4
    bg_thickness = min(h_cells, w_cells) // 8

    # Set foreground mask
    fg_mask_cells[fg_start_row:fg_end_row, fg_start_col:fg_end_col] = 1

    # Set background mask
    bg_mask_cells[:bg_thickness, :] = 1  # Top region
    bg_mask_cells[-bg_thickness:, :] = 1  # Bottom region
    bg_mask_cells[:, :bg_thickness] = 1  # Left region
    bg_mask_cells[:, -bg_thickness:] = 1  # Right region

    return fg_mask_cells, bg_mask_cells

fg_mask_cells, bg_mask_cells = initialize_masks(h_cells, w_cells)


graph_cut_result_cells = segment_image(
    im_cells, fg_mask_cells, bg_mask_cells, 
    unary_weight=1, pairwise_weight=1, n_bins=8)

# fig, axes = plt.subplots(1, 3, figsize=(9.5,5), sharey=True)
# axes[0].set_title('Initial foreground mask')
# axes[0].imshow(draw_mask_on_image(im_cells, fg_mask_cells))
# axes[1].set_title('Initial background mask')
# axes[1].imshow(draw_mask_on_image(im_cells, bg_mask_cells))
# axes[2].set_title('Graph cut result')
# axes[2].imshow(draw_mask_on_image(im_cells, graph_cut_result_cells))
# fig.tight_layout()


import skimage.data

im_bike = skimage.data.stereo_motorcycle()[0]
h_bike, w_bike = im_bike.shape[:2]
fg_mask_bike = np.zeros([h_bike, w_bike])
bg_mask_bike = np.zeros([h_bike, w_bike])

# Set some appropriate parts of fg_mask and bg_mask to 1 for initialization.
# YOUR CODE HERE
fg_mask_bike[150:400, 150:600] = 1
bg_mask_bike = 1 - fg_mask_bike
# bg_mask_bike[:h//4, :] = 1
# bg_mask_bike[3*h//4:, :] = 1
# bg_mask_bike[:, :w//4] = 1
# bg_mask_bike[:, 3*w//4:] = 1

graph_cut_result_bike = segment_image(
    im_bike, fg_mask_bike, bg_mask_bike, 
    unary_weight=1, pairwise_weight=1, n_bins=8)

# fig, axes = plt.subplots(1, 3, figsize=(9.5,5), sharey=True)
# axes[0].set_title('Initial foreground mask')
# axes[0].imshow(draw_mask_on_image(im_bike, fg_mask_bike))
# axes[1].set_title('Initial background mask')
# axes[1].imshow(draw_mask_on_image(im_bike, bg_mask_bike))
# axes[2].set_title('Graph cut result')
# axes[2].imshow(draw_mask_on_image(im_bike, graph_cut_result_bike))
# fig.tight_layout()

# POINTS: 4

def iterative_opt(im, fg_mask, n_bins, unary_weight,
                  pairwise_edges, pairwise_costs, n_iter):
    # YOUR CODE HERE
    for _ in range(n_iter): 
        fg_histogram = calculate_histogram(im, fg_mask, n_bins)
        bg_histogram = calculate_histogram(im, 1-fg_mask, n_bins)

        foreground_prob = foreground_pmap(im, fg_histogram, bg_histogram) 

        unary_fg = unary_potentials(foreground_prob, unary_weight)
        unary_bg = unary_potentials(1 - foreground_prob, unary_weight)

        height, width, _ = im.shape
        # pairwise_edges, pairwise_costs = pairwise_potentials(im, pairwise_weight)
        segmented_mask =  graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs)
        
        fg_mask = segmented_mask
    return segmented_mask

    # raise NotImplementedError()

labels_5 = iterative_opt(
    im, graph_cut_result, n_bins, unary_weight, pairwise_edges, pairwise_costs, n_iter=5)
labels_10 = iterative_opt(
    im, labels_5, n_bins, unary_weight, pairwise_edges, pairwise_costs, n_iter=5)

# fig, axes = plt.subplots(1, 3, figsize=(9.5,4), sharex=True, sharey=True)
# axes[0].set_title('Initial')
# axes[0].imshow(draw_mask_on_image(im, graph_cut_result))
# axes[1].set_title(f'After 5 iterations')
# axes[1].imshow(draw_mask_on_image(im, labels_5))
# axes[2].set_title(f'After 10 iterations')
# axes[2].imshow(draw_mask_on_image(im, labels_10))
# fig.tight_layout()