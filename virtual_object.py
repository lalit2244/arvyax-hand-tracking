import cv2
import numpy as np

class VirtualObject:
    """Virtual object/boundary on screen"""
    
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        
        # Define virtual boundaries
        self.boundaries = {
            "main": {
                "type": "rectangle",
                "points": [(int(screen_width * 0.6), int(screen_height * 0.3)),
                          (int(screen_width * 0.8), int(screen_height * 0.7))],
                "color": (0, 255, 0),  # Green
                "thickness": 3
            },
            "warning_zone": {
                "type": "rectangle",
                "points": [(int(screen_width * 0.55), int(screen_height * 0.25)),
                          (int(screen_width * 0.85), int(screen_height * 0.75))],
                "color": (0, 165, 255),  # Orange
                "thickness": 2,
                "dashed": True
            }
        }
    
    def draw(self, frame):
        """Draw all virtual boundaries on frame"""
        for boundary in self.boundaries.values():
            if boundary["type"] == "rectangle":
                pt1, pt2 = boundary["points"]
                
                if boundary.get("dashed", False):
                    # Draw dashed rectangle
                    x1, y1 = pt1
                    x2, y2 = pt2
                    
                    # Draw top line
                    for x in range(x1, x2, 15):
                        cv2.line(frame, (x, y1), (min(x+7, x2), y1), 
                                boundary["color"], boundary["thickness"])
                    
                    # Draw bottom line
                    for x in range(x1, x2, 15):
                        cv2.line(frame, (x, y2), (min(x+7, x2), y2), 
                                boundary["color"], boundary["thickness"])
                    
                    # Draw left line
                    for y in range(y1, y2, 15):
                        cv2.line(frame, (x1, y), (x1, min(y+7, y2)), 
                                boundary["color"], boundary["thickness"])
                    
                    # Draw right line
                    for y in range(y1, y2, 15):
                        cv2.line(frame, (x2, y), (x2, min(y+7, y2)), 
                                boundary["color"], boundary["thickness"])
                else:
                    cv2.rectangle(frame, pt1, pt2, 
                                boundary["color"], boundary["thickness"])
            
            elif boundary["type"] == "circle":
                cv2.circle(frame, boundary["center"], boundary["radius"],
                          boundary["color"], boundary["thickness"])
            
            elif boundary["type"] == "line":
                cv2.line(frame, boundary["points"][0], boundary["points"][1],
                        boundary["color"], boundary["thickness"])
        
        return frame
    
    def get_min_distance(self, point):
        """Calculate minimum distance from point to virtual boundary"""
        if point is None:
            return float('inf')
        
        px, py = point
        min_distance = float('inf')
        
        for boundary in self.boundaries.values():
            if boundary["type"] == "rectangle":
                x1, y1 = boundary["points"][0]
                x2, y2 = boundary["points"][1]
                
                # Calculate distance to rectangle edges
                if px < x1:
                    dx = x1 - px
                elif px > x2:
                    dx = px - x2
                else:
                    dx = 0
                
                if py < y1:
                    dy = y1 - py
                elif py > y2:
                    dy = py - y2
                else:
                    dy = 0
                
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < min_distance:
                    min_distance = distance
            
            elif boundary["type"] == "circle":
                cx, cy = boundary["center"]
                radius = boundary["radius"]
                
                distance = max(0, np.sqrt((px - cx)**2 + (py - cy)**2) - radius)
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else self.width  # Return screen width as max distance
    
    def is_inside_boundary(self, point):
        """Check if point is inside the main boundary"""
        if point is None:
            return False
        
        px, py = point
        boundary = self.boundaries["main"]
        
        if boundary["type"] == "rectangle":
            x1, y1 = boundary["points"][0]
            x2, y2 = boundary["points"][1]
            return x1 <= px <= x2 and y1 <= py <= y2
        
        elif boundary["type"] == "circle":
            cx, cy = boundary["center"]
            radius = boundary["radius"]
            return (px - cx)**2 + (py - cy)**2 <= radius**2
        
        return False