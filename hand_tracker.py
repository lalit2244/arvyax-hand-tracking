import cv2
import numpy as np
from collections import deque

class HandTracker:
    """Hand tracking using classical computer vision techniques"""
    
    def __init__(self):
        # Skin color ranges in HSV (optimized for various skin tones)
        self.lower_skin_hsv = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
        
        # Alternative skin range for different lighting
        self.lower_skin_hsv2 = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin_hsv2 = np.array([30, 200, 255], dtype=np.uint8)
        
        # YCrCb skin range (often more robust)
        self.lower_skin_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        self.upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
        
        # Tracking history for smoothing
        self.center_history = deque(maxlen=10)
        self.fingertip_history = deque(maxlen=10)
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        
        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Debug mode
        self.debug_mode = False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        # Resize for performance (keep aspect ratio)
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Apply slight Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return frame
    
    def detect_skin_hsv(self, frame):
        """Detect skin in HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask1 = cv2.inRange(hsv, self.lower_skin_hsv, self.upper_skin_hsv)
        mask2 = cv2.inRange(hsv, self.lower_skin_hsv2, self.upper_skin_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        return mask
    
    def detect_skin_ycrcb(self, frame):
        """Detect skin in YCrCb color space (often better for skin)"""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_skin_ycrcb, self.upper_skin_ycrcb)
        return mask
    
    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Threshold to get binary mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        return fg_mask
    
    def detect_hand_region(self, frame):
        """Main hand detection pipeline"""
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            height, width = processed_frame.shape[:2]
            
            # Get skin masks from different color spaces
            skin_mask_hsv = self.detect_skin_hsv(processed_frame)
            skin_mask_ycrcb = self.detect_skin_ycrcb(processed_frame)
            
            # Get motion mask
            motion_mask = self.detect_motion(processed_frame)
            
            # Combine masks (skin AND motion for better accuracy)
            combined_mask = cv2.bitwise_and(skin_mask_hsv, motion_mask)
            combined_mask = cv2.bitwise_or(combined_mask, skin_mask_ycrcb)
            
            # Apply morphological operations
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel_close)
            combined_mask = cv2.dilate(combined_mask, self.kernel_close, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, combined_mask
            
            # Filter contours by area and shape
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (too small or too large)
                if area < 500 or area > 50000:
                    continue
                
                # Filter by solidity (compactness)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity < 0.3:  # Too irregular
                        continue
                
                valid_contours.append(contour)
            
            if not valid_contours:
                return None, combined_mask
            
            # Select the largest valid contour (most likely hand)
            hand_contour = max(valid_contours, key=cv2.contourArea)
            
            # Additional validation: check aspect ratio
            x, y, w, h = cv2.boundingRect(hand_contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:  # Too elongated
                return None, combined_mask
            
            # Scale contour back to original size if needed
            if width != frame.shape[1]:
                scale_x = frame.shape[1] / width
                scale_y = frame.shape[0] / height
                hand_contour = hand_contour * np.array([scale_x, scale_y])
                hand_contour = hand_contour.astype(np.int32)
            
            self.detection_count += 1
            return hand_contour, combined_mask
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in hand detection: {e}")
            return None, np.zeros((100, 100), dtype=np.uint8)
    
    def get_hand_center(self, contour):
        """Calculate hand center point with smoothing"""
        if contour is None or len(contour) < 5:
            return None
        
        try:
            # Calculate moments
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
            
            # Calculate centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Apply smoothing using moving average
            self.center_history.append((cx, cy))
            
            # Calculate smoothed center
            if len(self.center_history) > 1:
                smoothed_x = int(np.mean([p[0] for p in self.center_history]))
                smoothed_y = int(np.mean([p[1] for p in self.center_history]))
                return (smoothed_x, smoothed_y)
            else:
                return (cx, cy)
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating hand center: {e}")
            return None
    
    def find_fingertips(self, contour):
        """Find fingertips using convex hull defects"""
        if contour is None or len(contour) < 50:
            return []
        
        try:
            # Simplify contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 4:
                return []
            
            # Get convex hull and defects
            hull = cv2.convexHull(approx, returnPoints=False)
            
            if len(hull) > 3:
                try:
                    defects = cv2.convexityDefects(approx, hull)
                except:
                    return []
                
                if defects is None:
                    return []
                
                fingertips = []
                
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    
                    # Calculate the angle
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                    
                    # Filter defects to find fingertips
                    if angle <= np.pi / 2 and d > 10000:  # 90 degrees
                        fingertips.append(far)
                
                # Remove duplicates (points too close together)
                unique_fingertips = []
                for tip in fingertips:
                    if not any(np.sqrt((tip[0] - u[0])**2 + (tip[1] - u[1])**2) < 20 
                              for u in unique_fingertips):
                        unique_fingertips.append(tip)
                
                # Smooth fingertip positions
                if unique_fingertips:
                    self.fingertip_history.append(unique_fingertips)
                    
                    # Apply smoothing if we have history
                    if len(self.fingertip_history) > 3:
                        smoothed_tips = []
                        for i in range(len(unique_fingertips)):
                            if i < len(unique_fingertips):
                                # Average last few positions for each fingertip
                                tip_history = []
                                for hist in self.fingertip_history:
                                    if i < len(hist):
                                        tip_history.append(hist[i])
                                
                                if tip_history:
                                    avg_x = int(np.mean([p[0] for p in tip_history]))
                                    avg_y = int(np.mean([p[1] for p in tip_history]))
                                    smoothed_tips.append((avg_x, avg_y))
                        
                        return smoothed_tips[:5]  # Return up to 5 fingertips
                
                return unique_fingertips[:5]
            
            return []
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error finding fingertips: {e}")
            return []
    
    def get_hand_bounding_box(self, contour):
        """Get bounding box around hand"""
        if contour is None:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, w, h)
    
    def get_hand_orientation(self, contour):
        """Calculate hand orientation using PCA"""
        if contour is None or len(contour) < 10:
            return 0
        
        try:
            # Reshape contour data
            data_pts = contour.reshape(-1, 2).astype(np.float32)
            
            # Perform PCA analysis
            mean = np.empty((0))
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
            
            # Calculate orientation angle
            angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
            return np.degrees(angle)
            
        except:
            return 0
    
    def calculate_hand_features(self, contour):
        """Calculate various hand features for better tracking"""
        if contour is None:
            return {}
        
        features = {}
        
        # Area
        features['area'] = cv2.contourArea(contour)
        
        # Perimeter
        features['perimeter'] = cv2.arcLength(contour, True)
        
        # Circularity
        if features['perimeter'] > 0:
            features['circularity'] = (4 * np.pi * features['area']) / (features['perimeter'] ** 2)
        else:
            features['circularity'] = 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        features['bounding_box'] = (x, y, w, h)
        features['aspect_ratio'] = w / h if h > 0 else 0
        
        # Convex hull area ratio
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = features['area'] / hull_area if hull_area > 0 else 0
        
        return features
    
    def reset_background_model(self):
        """Reset the background subtractor"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
    
    def get_detection_accuracy(self):
        """Get detection accuracy percentage"""
        if self.frame_count == 0:
            return 0
        return (self.detection_count / self.frame_count) * 100
    
    def enable_debug_mode(self, enabled=True):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
    
    def draw_debug_info(self, frame, contour, mask, hand_center, fingertips):
        """Draw debug information on frame"""
        debug_frame = frame.copy()
        
        # Draw contour
        if contour is not None:
            cv2.drawContours(debug_frame, [contour], -1, (0, 255, 255), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(debug_frame, [hull], -1, (255, 0, 255), 1)
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Draw hand center
        if hand_center:
            cv2.circle(debug_frame, hand_center, 8, (0, 0, 255), -1)
            cv2.circle(debug_frame, hand_center, 8, (255, 255, 255), 2)
        
        # Draw fingertips
        for fingertip in fingertips:
            cv2.circle(debug_frame, fingertip, 6, (255, 0, 0), -1)
            cv2.circle(debug_frame, fingertip, 6, (255, 255, 255), 1)
        
        # Draw mask in corner
        if mask is not None:
            mask_resized = cv2.resize(mask, (160, 120))
            mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            debug_frame[10:130, 10:170] = mask_color
            cv2.rectangle(debug_frame, (10, 10), (170, 130), (255, 255, 255), 1)
            cv2.putText(debug_frame, "Skin Mask", (15, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add text information
        info_text = f"Detections: {self.detection_count}/{self.frame_count}"
        cv2.putText(debug_frame, info_text, (180, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        if contour is not None:
            area = cv2.contourArea(contour)
            area_text = f"Area: {int(area)}"
            cv2.putText(debug_frame, area_text, (180, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return debug_frame

# Test function
if __name__ == "__main__":
    # Simple test
    print("Testing HandTracker class...")
    
    tracker = HandTracker()
    tracker.enable_debug_mode(True)
    
    # Create a test image (simulating hand)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image.fill(100)
    
    # Draw a simple hand shape
    cv2.circle(test_image, (320, 240), 80, (180, 150, 130), -1)  # Palm
    cv2.circle(test_image, (320, 140), 20, (180, 150, 130), -1)  # Finger
    
    # Test detection
    contour, mask = tracker.detect_hand_region(test_image)
    center = tracker.get_hand_center(contour)
    fingertips = tracker.find_fingertips(contour)
    
    print(f"Contour found: {contour is not None}")
    print(f"Hand center: {center}")
    print(f"Fingertips found: {len(fingertips)}")
    
    if contour is not None:
        features = tracker.calculate_hand_features(contour)
        print(f"Hand area: {features.get('area', 0):.0f}")
        print(f"Detection accuracy: {tracker.get_detection_accuracy():.1f}%")
    
    print("\nHandTracker test completed successfully!")
