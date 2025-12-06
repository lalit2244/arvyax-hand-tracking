import cv2
import numpy as np
from collections import Counter

class StateManager:
    """Manages state classification and visual feedback"""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # State thresholds (in pixels)
        self.SAFE_DISTANCE = 150
        self.WARNING_DISTANCE = 50
        self.DANGER_DISTANCE = 20
        
        # State colors
        self.STATE_COLORS = {
            "SAFE": (0, 255, 0),      # Green
            "WARNING": (0, 165, 255),  # Orange
            "DANGER": (0, 0, 255)      # Red
        }
        
        # State tracking
        self.current_state = "SAFE"
        self.state_history = []
        self.max_history = 10
        
        # Performance tracking
        self.frame_times = []
        self.fps = 0
    
    def classify_state(self, distance):
        """Classify state based on distance to boundary"""
        if distance == float('inf'):
            return "SAFE"  # No hand detected
        
        if distance > self.SAFE_DISTANCE:
            return "SAFE"
        elif distance > self.WARNING_DISTANCE:
            return "WARNING"
        else:
            return "DANGER"
    
    def update_state(self, distance):
        """Update current state with hysteresis for stability"""
        new_state = self.classify_state(distance)
        
        # Add to history
        self.state_history.append(new_state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Use mode of recent history for stability
        if len(self.state_history) >= 3:
            mode_state = Counter(self.state_history).most_common(1)[0][0]
            # Only transition to DANGER if consistent
            if self.current_state != "DANGER" or mode_state == "DANGER":
                self.current_state = mode_state
        else:
            self.current_state = new_state
        
        return self.current_state
    
    def draw_state_overlay(self, frame, distance, hand_center):
        """Draw state information on frame"""
        # Draw state panel
        panel_height = 120
        cv2.rectangle(frame, (0, 0), (self.screen_width, panel_height),
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (0, 0), (self.screen_width, panel_height),
                     (100, 100, 100), 2)
        
        # Draw current state
        state_color = self.STATE_COLORS[self.current_state]
        state_text = f"STATE: {self.current_state}"
        
        # For DANGER state, make it more prominent
        if self.current_state == "DANGER":
            # Pulsing effect
            pulse = int(50 * (np.sin(cv2.getTickCount() * 0.01) + 1))
            danger_color = (0, 0, 255)
            
            # Draw danger background
            cv2.rectangle(frame, (0, panel_height), 
                         (self.screen_width, panel_height + 60),
                         (0, 0, pulse), -1)
            
            # Draw "DANGER DANGER" text
            danger_text = "DANGER DANGER"
            text_size = cv2.getTextSize(danger_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       1.5, 3)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(frame, danger_text, (text_x, panel_height + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, danger_text, (text_x, panel_height + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, danger_color, 2)
        
        # Draw state in top panel
        cv2.putText(frame, state_text, (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, state_text, (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, state_color, 1)
        
        # Draw distance information (handle infinity case)
        if distance == float('inf'):
            dist_text = "Distance: No hand detected"
        else:
            dist_text = f"Distance: {int(distance)}px"
        
        cv2.putText(frame, dist_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw thresholds
        thresholds_text = f"Thresholds: SAFE>{self.SAFE_DISTANCE} | WARNING>{self.WARNING_DISTANCE} | DANGER<{self.DANGER_DISTANCE}"
        cv2.putText(frame, thresholds_text, (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw distance visualization if hand is detected
        if hand_center and distance != float('inf'):
            # Draw line from hand to nearest boundary point
            boundary_center = (int(self.screen_width * 0.7), 
                             int(self.screen_height * 0.5))
            cv2.line(frame, hand_center, boundary_center, state_color, 2)
            
            # Draw distance arc
            radius = min(int(distance), 200)
            cv2.circle(frame, hand_center, radius, state_color, 2)
            
            # Draw hand center
            cv2.circle(frame, hand_center, 10, state_color, -1)
            cv2.circle(frame, hand_center, 10, (255, 255, 255), 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (self.screen_width - 120, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def update_fps(self, frame_time):
        """Update FPS calculation"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 0:
            avg_time = np.mean(self.frame_times)
            self.fps = 1000 / avg_time if avg_time > 0 else 0