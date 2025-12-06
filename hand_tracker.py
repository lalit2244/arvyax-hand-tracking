import cv2
import numpy as np

class HandTracker:
    """Hand tracking using classical computer vision techniques"""
    
    def __init__(self):
        # Skin color ranges in HSV (tuned for various skin tones)
        self.lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Alternative skin range for different lighting
        self.lower_skin2 = np.array([0, 50, 0], dtype=np.uint8)
        self.upper_skin2 = np.array([120, 150, 70], dtype=np.uint8)
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
        
        # Tracking
        self.prev_center = None
        
    def detect_skin_ycrcb(self, frame):
        """Detect skin in YCrCb color space (often better for skin detection)"""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Skin color range in YCrCb
        lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        
        mask = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        return mask
    
    def detect_skin_hsv(self, frame):
        """Detect skin in HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def detect_hand_region(self, frame):
        """Main hand detection pipeline"""
        # Resize for performance
        original_height, original_width = frame.shape[:2]
        scale_factor = 0.5
        width = int(original_width * scale_factor)
        height = int(original_height * scale_factor)
        small_frame = cv2.resize(frame, (width, height))
        
        # Get skin masks from different color spaces
        skin_mask_hsv = self.detect_skin_hsv(small_frame)
        skin_mask_ycrcb = self.detect_skin_ycrcb(small_frame)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
        
        # Apply Gaussian blur
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Threshold
        _, skin_mask = cv2.threshold(skin_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, skin_mask
        
        # Find the largest contour (likely hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours
        if cv2.contourArea(largest_contour) < 500:  # Minimum area threshold
            return None, skin_mask
        
        # Scale contour back to original size
        largest_contour = largest_contour / scale_factor
        largest_contour = largest_contour.astype(np.int32)
        
        return largest_contour, skin_mask
    
    def get_hand_center(self, contour):
        """Calculate hand center point"""
        if contour is None or len(contour) < 5:
            return None
        
        try:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Simple smoothing
                if self.prev_center:
                    cx = int(self.prev_center[0] * 0.3 + cx * 0.7)
                    cy = int(self.prev_center[1] * 0.3 + cy * 0.7)
                
                self.prev_center = (cx, cy)
                return (cx, cy)
        except:
            pass
        
        return None