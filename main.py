from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import cv2
import os

def fill_mask_with_nearest_neighbor(img_lab, mask):
    known = (mask == 0)
    dist, inds = distance_transform_edt(~known, return_indices=True)
    filled = img_lab[inds[0], inds[1]]
    return filled

def region_filling_algorithm(img_path, mask_path, patch_size=9, output="result.png"):  
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    mask = (mask > 127).astype(np.uint8) * 255 

    # Convert image to LAB color space (for better patch matching)
    img_lab = rgb2lab(img)
    img_lab_filled = fill_mask_with_nearest_neighbor(img_lab, mask)

    # Working image and mask
    workimg = img_lab_filled.copy()
    workmask = mask.copy()
    
    # Confidence map (1 for source region, 0 for target region)
    confidence = (workmask == 0).astype(float)
    
    fill_step = 0
    
    while np.sum(workmask) > 0:
        fill_step += 1

        # Contour (fill front)
        contour = find_fill_front(workmask)

        # Priorities for all points on the fill front
        priorities, confidences_front = compute_priorities(workimg, workmask, confidence, contour, patch_size)
        
        # Get the point with highest priority
        p_idx = np.argmax(priorities)
        p = contour[p_idx]
        confidence_p = confidences_front[p_idx]
        
        # Find the best matching patch from source region
        best_patch = find_best_exemplar(workimg, workmask, p, patch_size)
        
        # Copy best patch data to target region
        workimg, workmask, confidence = update_image(workimg, workmask, confidence, p, best_patch, patch_size, confidence_p)
        
        print(f"Fill step {fill_step}, {np.sum(workmask > 0)} pixels remaining")
    
    # Convert final result back to RGB
    filled_img = lab2rgb(workimg)
    filled_img = np.clip(filled_img * 255, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output, cv2.cvtColor(filled_img, cv2.COLOR_RGB2BGR))
    
    return filled_img

def find_fill_front(mask):
    # Dilate the source region
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(255 - mask, kernel)
    
    # Get fill front
    fill_front = dilated & mask
    
    # Get coordinates of fill front
    contour = np.where(fill_front > 0)
    contour = list(zip(contour[0], contour[1]))
    
    return contour

def compute_priorities(img, mask, confidence, contour, patch_size):
    h, w = mask.shape[:2]
    half_patch = patch_size // 2
    priorities = []
    C_list = []
    grad_y, grad_x = np.gradient(img[:,:,0])
    
    for p in contour:
        y, x = p
        
        # Skip points too close to the border
        if (y < half_patch or y >= h - half_patch or 
            x < half_patch or x >= w - half_patch):
            priorities.append(0)
            C_list.append(0)
            continue
        
        # Compute confidence term
        patch_conf = confidence[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
        patch_mask = mask[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1] == 0
        C_p = np.sum(patch_conf * patch_mask) / (patch_size * patch_size)
        
        # Calculate normal to the contour
        normal = compute_normal(mask, p)
        
        # Calculate isophote (gradient perpendicular to the edge)
        isophote_y = grad_y[y, x]
        isophote_x = grad_x[y, x]
        isophote = np.array([isophote_y, isophote_x])
        
        # Calculate data term
        alpha = 100.0
        D_p = abs(np.dot(isophote, normal)) / alpha
        
        # Calculate priority
        priority = C_p * D_p
        priorities.append(priority)
 
        C_list.append(C_p)
    
    return np.array(priorities), np.array(C_list)

def compute_normal(mask, point):
    y, x = point

    # Sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Patch around the point
    h, w = mask.shape
    pad = 1
    patch = mask[max(0, y-pad):min(h, y+pad+1), max(0, x-pad):min(w, x+pad+1)]
    
    # Deal with edge case
    sy, sx = patch.shape
    if sy < 3 or sx < 3:
        return np.array([0, -1])
    
    # Gradients
    grad_x = np.sum(patch * sobel_x[:sy, :sx])
    grad_y = np.sum(patch * sobel_y[:sy, :sx])
    
    # Normal (perpendicular to gradient)
    normal = np.array([-grad_y, -grad_x])
    
    # Normalize
    norm = np.sqrt(np.sum(normal**2))
    if norm > 0:
        normal = normal / norm
    
    return normal

def find_best_exemplar(img, mask, point, patch_size):
    y, x = point
    h, w = img.shape[:2]
    half_patch = patch_size // 2
    
    # Make sure point is not too close to border
    y = max(half_patch, min(y, h - half_patch - 1))
    x = max(half_patch, min(x, w - half_patch - 1))
    
    # Extract the template patch centered at the point
    template = img[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
    template_mask = mask[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1] == 0
    
    min_dist = float('inf')
    best_patch = (0, 0)
    
    # Get search region
    search_area = []
    for sy in range(half_patch, h - half_patch):
        for sx in range(half_patch, w - half_patch):
            if np.sum(mask[sy-half_patch:sy+half_patch+1, sx-half_patch:sx+half_patch+1]) == 0:
                search_area.append((sy, sx)) 
    
    for sy, sx in search_area:
        # Candidate patch
        candidate = img[sy-half_patch:sy+half_patch+1, sx-half_patch:sx+half_patch+1]
        
        # Compute SSD (Sum of Squared Differences) for known pixels
        diff = (template - candidate) ** 2
        mask_sum = np.sum(template_mask)
        dist = np.sum(diff * template_mask[:,:,np.newaxis]) / mask_sum
        
        if dist < min_dist:
            min_dist = dist
            best_patch = (sy, sx)
    
    return best_patch

def update_image(img, mask, confidence, point, best_patch, patch_size, confidence_p):
    y, x = point
    by, bx = best_patch
    half_patch = patch_size // 2
    
    h, w = img.shape[:2]

    # Deal with edge case 
    y = max(half_patch, min(y, h - half_patch - 1))
    x = max(half_patch, min(x, w - half_patch - 1))
    by = max(half_patch, min(by, h - half_patch - 1))
    bx = max(half_patch, min(bx, w - half_patch - 1))
    
    # Target patches
    target_y_start = y - half_patch
    target_y_end = y + half_patch + 1
    target_x_start = x - half_patch
    target_x_end = x + half_patch + 1
    
    # Source patches
    source_y_start = by - half_patch
    source_y_end = by + half_patch + 1
    source_x_start = bx - half_patch
    source_x_end = bx + half_patch + 1
    
     # Masks for target patch
    target_mask = mask[target_y_start:target_y_end, target_x_start:target_x_end]
    fill_mask = target_mask > 0
    
    # Copy data from source to target for unknown pixels
    img[target_y_start:target_y_end, target_x_start:target_x_end][fill_mask] = \
        img[source_y_start:source_y_end, source_x_start:source_x_end][fill_mask]
    
    mask[target_y_start:target_y_end, target_x_start:target_x_end] = 0
    
    # Update confidence
    confidence[target_y_start:target_y_end, target_x_start:target_x_end][fill_mask] = confidence_p
    
    return img, mask, confidence

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Region-filling Algorithm')
    parser.add_argument('-i', '--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('-m', '--mask', type=str, required=True, help='Path to the mask image')
    parser.add_argument('-p', '--patch-size', type=int, default=9, help='Patch size (must be odd)')
    parser.add_argument('-o', '--output', type=str, default='result.png', help='Output path and filename')
    
    args = parser.parse_args()
    
    # Ensure patch size is odd
    if args.patch_size % 2 == 0:
        args.patch_size += 1
        print(f"Patch size must be odd. Using {args.patch_size} instead.")
    
    # Valid patch size
    img = cv2.imread(args.img)
    h, w = img.shape[:2]
    max_patch_size = min(h, w) - 2
    
    if args.patch_size > max_patch_size:
        orig_size = args.patch_size
        args.patch_size = max_patch_size if max_patch_size % 2 == 1 else max_patch_size - 1
        print(f"Patch size {orig_size} is too large for this image. Using {args.patch_size} instead.")
    
    output_dir = os.path.dirname(args.output)
    output_filename = os.path.basename(args.output)

    if not output_dir:
        output_dir = os.getcwd()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filled_img = region_filling_algorithm(
        args.img, 
        args.mask, 
        args.patch_size, 
        args.output
    )
    
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()