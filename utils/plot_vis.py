from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import cv2

def plot_segmentation_map(sample):
    """
    Plots the image with the labeled segmentation maps overlaid using PIL, with different colors for each segment.

    Parameters:
    - sample (dict): A dictionary containing 'image_path', 'segmentation_map', and 'label_name'.
    """
    # Load the image
    image_path = sample['image_path']
    image = Image.open(image_path).convert("RGB")

    # Create a mask image initialized to transparent
    mask = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    # Define colors for each label
    colors = [
        (255, 0, 0, 128),   # Red
        (0, 255, 0, 128),   # Green
        (0, 0, 255, 128),   # Blue
        (255, 255, 0, 128), # Yellow
        (0, 255, 255, 128), # Cyan
        (255, 0, 255, 128), # Magenta
    ]

    # Iterate over each segmentation in the segmentation map
    for idx, [label, segments] in enumerate(sample['segmentation_data']):
        color = colors[int(label) % len(colors)]  # Cycle through colors
        for segment in segments:
            # Polygon points must be in the form of a list of tuples
            polygon = [point for point in segment]
            draw.polygon(polygon, outline=color, fill=color)  # Semi-transparent color
            polygonx = [segment[i] for i in range(0, len(segment), 2)]
            polygony = [segment[i+1] for i in range(0, len(segment), 2)]

            # Annotate the label near the segment
            center_x = sum(x for x in polygonx) // len(polygonx)
            center_y = sum(y for y in polygony) // len(polygony)
            draw.text((center_x, center_y), sample['catIds'][label], fill=(0, 0, 0, 255))

    # Composite the mask with the original image
    overlay = Image.alpha_composite(image.convert('RGBA'), mask)

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Label: {sample['class_name']}")
    plt.axis('off')
    plt.show()
    plt.savefig('segmentation_map.png')

def plot_clip_scores_histogram(dataset, texts, check_attribute=None, concept_to_check=None,return_mean_scores=True):
    all_scores = {text: {'present': {}, 'absent': {}, 'present and absent': {}, 'inpainted': {}} for text in texts}
    
    for text in texts:
        for feature in check_attribute:
            for sample in dataset.List_data_image:
                label_name = sample['class_name']
                if feature == 'all':    
                    presence = 'present and absent'
                elif feature == 'present':
                    if concept_to_check not in sample.get('concepts', []):
                        continue
                    presence = 'present'
                elif feature == 'absent':
                    if concept_to_check in sample.get('concepts', []):
                        continue
                    presence = 'absent'
                elif feature == 'inpainted':
                    if concept_to_check not in sample.get('concepts', []):
                        continue
                    presence = 'inpainted'
                    all_scores[text].setdefault(presence, {}).setdefault(label_name, []).append(sample['clip_scores_inpaint'][text])
                    continue
                
                all_scores[text].setdefault(presence, {}).setdefault(label_name, []).append(sample['clip_scores'][text])
    
    all_clip_scores = [score for label_scores in all_scores.values() for presence_scores in label_scores.values() for scores in presence_scores.values() for score in scores]
    bins = np.linspace(min(all_clip_scores), max(all_clip_scores), 20)
    
    plt.figure(figsize=(15, 10))
    for text, label_scores in all_scores.items():
        for presence, label_data in label_scores.items():
            for label_name, scores in label_data.items():
                plt.hist(scores, bins=bins, alpha=0.5, label=f'{text} (Label {label_name}, {presence})')
                print(f"Text: {text}, Label: {label_name}, Presence: {presence} - Mean: {np.mean(scores):.4f}, Std Dev: {np.std(scores):.4f}")
    
    plt.title('Histogram of CLIP Scores by Label and Concept Presence')
    plt.xlabel('CLIP Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    plt.savefig('histogram.png')

    if return_mean_scores:
        dict_mean_scores = {}
        for text, label_scores in all_scores.items():
            for presence, label_data in label_scores.items():
                for label_name, scores in label_data.items():
                    print(presence)
                    if "and" in presence:
                        continue  # Skip combined presence categories
                    dict_mean_scores_key = f"{text}_{presence}"
                    dict_mean_scores[dict_mean_scores_key] = {}
                    dict_mean_scores[dict_mean_scores_key]['mean'] = np.mean(scores)
                    dict_mean_scores[dict_mean_scores_key]['variance'] = np.var(scores)
                    dict_mean_scores[dict_mean_scores_key]['list_scores'] = scores
        return dict_mean_scores
    
def plot_diff_clip_scores_histogram(dataset, texts, concept_to_check=None):
    all_scores ={text:[] for text in texts}
    all_impainted_area = []
    max_area = 0

    def compute_polygon_area(segmentation, sample):
        # Open the image to get its original size
        with Image.open(sample['image_path']) as img:
            orig_width, orig_height = img.size
        
        # Convert the segmentation list into a NumPy array of shape (n, 2)
        polygon = np.array(segmentation).reshape(-1, 2)
        
        # Scale coordinates to 224x224
        polygon[:, 0] = polygon[:, 0] * (224 / orig_width)
        polygon[:, 1] = polygon[:, 1] * (224 / orig_height)
        
        # Shoelace formula to calculate the area of the polygon
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return area

    for text in texts:
        for i,sample in enumerate(dataset.List_data_image):
            if concept_to_check not in sample.get('concepts', []):
                continue
            all_scores[text].append(sample['clip_scores'][text]-sample['clip_scores_inpaint'][text])
            
            # Compute area of impainted pixels thanks to 'segmentation_data'
            impainted_area = 0
            coco = COCO(sample['json_class_pth'])
            for label, segments in sample['segmentation_data']:
                if coco.loadCats(int(label))[0]["name"] != concept_to_check :
                    continue
                for segment in segments:
                    impainted_area += compute_polygon_area(segment, sample)

                                # Track the sample with the maximum impainted area
                if impainted_area > max_area:
                    max_area = impainted_area
                    max_area_path = sample['image_path']
                    print(i)

            all_impainted_area.append(impainted_area)

    print(max_area_path, max_area)

    # Plot area vs clip score
    plt.figure(figsize=(10, 5))
    plt.scatter(all_impainted_area, all_scores[text], alpha=0.5)
    plt.xlabel('Area of impainted pixels')
    plt.ylabel('Difference between original and inpainted CLIP score')
    plt.show()
    plt.savefig('area_vs_clip_score.png')
    
    '''all_clip_scores = [score for label_scores in all_scores.values() for score in label_scores]
    bins = np.linspace(min(all_clip_scores), max(all_clip_scores), 20)
    
    plt.figure(figsize=(15, 10))
    for text, label_scores in all_scores.items():
        plt.hist(label_scores, bins=bins, alpha=0.5, label=f'{text}')
        print(f"Text: {text} - Mean: {np.mean(label_scores):.4f}, Std Dev: {np.std(label_scores):.4f}")
    
    plt.title('Histogram of CLIP Scores by Label and Concept Presence')
    plt.xlabel('CLIP Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()'''

def print_sample_with_highest_clip_score(dataset, concept):
    highest_score = -1
    best_sample = None

    for sample in dataset.List_data_image:
        if sample['clip_scores_inpaint'] and concept in sample['clip_scores_inpaint']:
            score = sample['clip_scores_inpaint'][concept]
            if score > highest_score:
                highest_score = score
                best_sample = sample

    if best_sample:
        print(f"Sample with the highest CLIP score for '{concept}':")
        print(f"Image Path: {best_sample['image_path']}")
        print(f"Label Number: {best_sample['label_number']}")
        print(f"CLIP Score: {highest_score:.4f}")
    else:
        print(f"No samples found with CLIP scores for '{concept}'.")

def plot_heatmap(attention_map, save_path, title, image_path=None):
    # Normalize the attention map
    v = attention_map
    min_ = attention_map.min()
    max_ = attention_map.max()
    v = v - min_
    v = np.uint8((v / (max_ - min_)) * 255)

    # Apply color map
    heatmap = cv2.applyColorMap(v, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if image_path is not None:
        # Load the image
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        # Resize the heatmap to match the image dimensions
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Overlay the heatmap on the image
        superimposed_img = heatmap * 0.5 + img * 0.5
        superimposed_img = np.uint8(superimposed_img)

        # Plot the superimposed image
        plt.imshow(superimposed_img)
    else:
        # Plot the raw heatmap
        plt.imshow(heatmap)

    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    
def plot_heatmap_(heatmap, title,save_path='best_heatmap.png'):
    """
    Plot a heatmap.

    Parameters:
    heatmap (numpy.ndarray): The heatmap to plot.
    title (str): The title of the plot.
    """

    plt.imshow(heatmap, cmap='hot')
    plt.axis('off')
    plt.title(title)
    plt.colorbar()
    # Save the mosaic
    plt.savefig(save_path)

# Example usage
# sample = {
#     'image_path': 'path/to/image.jpg',
#     'segmentation_map': {1: [[(10, 10), (50, 50), (30, 30)]], 2:
