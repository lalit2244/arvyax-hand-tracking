import cv2
import numpy as np
import time
import threading
import queue
import sys
import os

class CameraManager:
    """Robust camera management with multiple fallback options"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.camera_index = 0
        self.camera = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.camera_thread = None
        self.last_frame = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Camera backends to try (Windows specific)
        self.backends_windows = [
            cv2.CAP_DSHOW,    # DirectShow (best for Windows)
            cv2.CAP_MSMF,     # Microsoft Media Foundation
            cv2.CAP_ANY       # Auto-detect
        ]
        
        # Camera backends for Linux/Mac
        self.backends_linux = [
            cv2.CAP_V4L2,     # Video4Linux2
            cv2.CAP_ANY
        ]
        
    def detect_available_cameras(self):
        """Detect available cameras with different indices"""
        available = []
        max_cameras = 5  # Check up to 5 camera indices
        
        print("🔍 Scanning for available cameras...")
        
        for i in range(max_cameras):
            cap = self._try_open_camera(i)
            if cap is not None:
                available.append(i)
                cap.release()
                print(f"  ✅ Found camera at index {i}")
                time.sleep(0.1)
        
        return available
    
    def _try_open_camera(self, index):
        """Try to open camera with multiple backends"""
        if sys.platform.startswith('win'):
            backends = self.backends_windows
        else:
            backends = self.backends_linux
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap
                    else:
                        cap.release()
            except:
                continue
        
        return None
    
    def start(self, camera_index=0):
        """Start camera capture thread"""
        if self.running:
            self.stop()
        
        self.camera_index = camera_index
        self.camera = self._try_open_camera(camera_index)
        
        if self.camera is None:
            print(f"❌ Could not open camera at index {camera_index}")
            print("📱 Switching to demo mode...")
            return False
        
        self.running = True
        self.camera_thread = threading.Thread(target=self._capture_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        print(f"✅ Camera started at index {camera_index}")
        return True
    
    def _capture_loop(self):
        """Camera capture loop running in separate thread"""
        self.frame_count = 0
        self.start_time = time.time()
        
        while self.running and self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Flip horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Resize if needed
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    self.last_frame = frame.copy()
                    
                    # Calculate FPS
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    # Put frame in queue (non-blocking)
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass  # Skip frame if queue is full
                
                else:
                    # Camera disconnected or error
                    print("⚠️ Camera read error, trying to reconnect...")
                    time.sleep(0.1)
                    self._reconnect_camera()
                    
            except Exception as e:
                print(f"⚠️ Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _reconnect_camera(self):
        """Try to reconnect to camera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        time.sleep(0.5)
        self.camera = self._try_open_camera(self.camera_index)
        
        if self.camera is None:
            print("❌ Camera reconnection failed")
            self.running = False
    
    def get_frame(self):
        """Get latest frame from camera"""
        try:
            # Try to get frame from queue (non-blocking)
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Return last frame if available
        if self.last_frame is not None:
            return self.last_frame
        
        # Create a placeholder frame
        return self._create_placeholder_frame()
    
    def _create_placeholder_frame(self):
        """Create a placeholder frame when camera is not available"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame.fill(40)  # Dark gray
        
        # Add text
        text = "NO CAMERA"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 2, (0, 0, 255), 3)
        
        # Add instructions
        info = "Check camera connection"
        info_size = cv2.getTextSize(info, font, 0.7, 1)[0]
        info_x = (self.width - info_size[0]) // 2
        cv2.putText(frame, info, (info_x, text_y + 50), font, 0.7, (200, 200, 200), 1)
        
        return frame
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=2.0)
            self.camera_thread = None
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        print("🛑 Camera stopped")
    
    def is_running(self):
        """Check if camera is running"""
        return self.running and self.camera is not None
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def get_camera_info(self):
        """Get camera information"""
        if self.camera is None:
            return "Camera not available"
        
        info = {
            "index": self.camera_index,
            "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.fps,
            "frame_count": self.frame_count
        }
        
        return info

# Test the camera manager
if __name__ == "__main__":
    print("Testing Camera Manager...")
    print("=" * 50)
    
    # Create camera manager
    cam_manager = CameraManager()
    
    # Detect available cameras
    available_cams = cam_manager.detect_available_cameras()
    
    if available_cams:
        print(f"\n📊 Available cameras: {available_cams}")
        print(f"Starting camera at index {available_cams[0]}...")
        
        # Start camera
        if cam_manager.start(available_cams[0]):
            print("\n🎥 Camera streaming started!")
            print("Press 'q' to quit, 's' to take screenshot\n")
            
            try:
                screenshot_count = 0
                
                while True:
                    # Get frame
                    frame = cam_manager.get_frame()
                    
                    # Display frame
                    cv2.imshow("Camera Test", frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Take screenshot
                        screenshot_count += 1
                        filename = f"screenshot_{screenshot_count}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 Screenshot saved: {filename}")
                    
                    # Display info
                    info = cam_manager.get_camera_info()
                    if isinstance(info, dict):
                        fps_text = f"FPS: {info['fps']:.1f}"
                        cv2.putText(frame, fps_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            finally:
                cam_manager.stop()
                cv2.destroyAllWindows()
    else:
        print("\n❌ No cameras detected!")
        print("\nTroubleshooting steps:")
        print("1. Check if camera is connected")
        print("2. Make sure no other app is using the camera")
        print("3. Try running as administrator")
        print("4. Update camera drivers")
        print("\nCreating demo frame instead...")
        
        # Show placeholder frame
        frame = cam_manager._create_placeholder_frame()
        cv2.imshow("Camera Test (Demo Mode)", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    print("\n✅ Camera Manager test completed!")
