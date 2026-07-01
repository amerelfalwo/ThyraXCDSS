import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def create_final_ultrasound_image(
    original_path: str,
    mask_path: str,
    bbox: list[int],
    final_label: str,
    output_path: str
) -> str:
    """
    Composites the original ultrasound image with the segmentation mask, bounding box, and classification label.

    Args:
        original_path: Path to the original ultrasound image.
        mask_path: Path to the segmentation mask image.
        bbox: A list containing [x, y, w, h] representing the bounding box.
        final_label: The final classification label text (e.g., 'Malignant').
        output_path: Path where the resulting composited image should be saved.

    Returns:
        The path to the saved composited image (output_path).
    """
    try:
        # 1. Read images
        original = cv2.imread(original_path)
        if original is None:
            raise ValueError(f"Could not read original image at {original_path}")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask image at {mask_path}")

        # Ensure both images have the same dimensions just in case
        if original.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Mask Blending
        # Create a green overlay image (same shape as original)
        green_overlay = np.zeros_like(original)
        green_overlay[:] = (0, 255, 0)  # BGR format: Green

        # Create a boolean mask where pixel values are > 127
        mask_boolean = mask > 127

        # Alpha for blending
        alpha = 0.4
        
        # Apply blending only on the mask area
        blended = original.copy()
        
        # Extract regions
        original_region = blended[mask_boolean]
        green_region = green_overlay[mask_boolean]
        
        # Calculate blended pixels: output_pixel = alpha * green + (1 - alpha) * original_pixel
        blended[mask_boolean] = cv2.addWeighted(green_region, alpha, original_region, 1 - alpha, 0)

        # 3. Bounding Box
        x, y, w, h = bbox
        
        # Color: Dark purple BGR (130, 0, 75)
        bbox_color = (130, 0, 75)
        bbox_thickness = 2
        
        cv2.rectangle(blended, (x, y), (x + w, y + h), bbox_color, bbox_thickness)

        # 4. Label Drawing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White

        # Calculate text size to dynamically size the background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(final_label, font, font_scale, font_thickness)
        
        # Define padding for the text background
        padding = 5
        
        # Start coordinates for the text background (just above the bounding box)
        # Ensure it doesn't go outside the top of the image
        bg_y1 = max(0, y - text_height - (padding * 2) - baseline)
        bg_y2 = y
        bg_x1 = x
        bg_x2 = x + text_width + (padding * 2)
        
        # Ensure x2 does not exceed image width
        bg_x2 = min(blended.shape[1], bg_x2)

        # Draw solid background rectangle for text
        cv2.rectangle(blended, (bg_x1, bg_y1), (bg_x2, bg_y2), bbox_color, cv2.FILLED)

        # Calculate text position (bottom-left corner of the text string in the image)
        text_x = bg_x1 + padding
        text_y = bg_y2 - padding - baseline // 2
        
        # Draw the text
        cv2.putText(blended, final_label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # 5. Save the blended image
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        cv2.imwrite(output_path, blended)
        
        logger.info(f"Successfully saved composited image to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create composited image: {e}")
        raise
