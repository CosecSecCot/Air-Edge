import cv2
import numpy as np
import time
from pygame.math import clamp
from pyglm import glm
from application.Puck import Puck
from application.mallet import Mallet
from utils import logger
import pygame


class Application:
    __screen_width = 540
    __screen_height = 960

    def __init__(self) -> None:
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode(
            (self.__screen_width, self.__screen_height),
            pygame.FULLSCREEN
        )
        self.clock = pygame.time.Clock()
        self.delta_time = 0.1
        self.running = True

        self.puck = Puck()
        self.puck.pos.x = self.__screen_width // 2
        self.puck.pos.y = self.__screen_height * 2 // 3

        self.mallet = Mallet()

        self.sound = pygame.mixer.music.load("src/application/pick.mp3")

        self.cap = cv2.VideoCapture(1)
        # === Global State ===
        self.pts = []                      # User-clicked polygon points
        self.selection_complete = False    # True after 4 clicks
        self.kf = None                     # Kalman filter
        self.dt = 1/30.0                   # Assume 30 FPS for Kalman timing
        self.initialized = False           # True once tracking is set
        self.roi_hist = None               # Color histogram for back projection
        self.track_window = None           # Current tracking window
        self.object_lost = False           # Flag for lost object tracking
        self.lost_counter = 0              # Counter for frames with lost object
        self.lost_threshold = 10           # Frames to wait before declaring lost
        self.max_lost_time = 100           # Max frames to keep lost object for re-tracking
        self.last_seen_frame = 0           # Last frame index when object was seen
        self.frame_index = 0               # Current frame index
        self.object_features = None        # Stored features for re-identification
        
        # self.Cropping variables
        self.crop_mode = False             # True when in crop mode
        self.cropping = False              # True while drawing crop rectangle
        self.crop_start = None             # Starting point of crop rectangle
        self.crop_end = None               # Ending point of crop rectangle
        self.cropped_view = False          # True when viewing cropped area
        self.crop_rect = None              # Stored crop rectangle (x, y, w, h)
        self.show_selection = True         # Show selection points during drawing, hide after initialization

    def __del__(self) -> None:
        logger.client_logger.info("Application Closed!")
        # self.network_client.close()
        pygame.mixer.quit()
        pygame.quit()
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        logger.client_logger.info("Application Started!")

        # === Main ===
        cap = cv2.VideoCapture(1)
        cv2.namedWindow("Multi-Feature Tracker")
        cv2.setMouseCallback("Multi-Feature Tracker", self.on_mouse)
        
        # Build Gabor filters once at startup
        gabor_filters = self.build_gabor_filters()
        logger.client_logger.debug(f"Created {len(gabor_filters)} Gabor filters for texture analysis")
        
        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0

        while (self.running):
            ret, frame = cap.read()
            if not ret:
                self.running = False
                break
            
            self.frame_index += 1
            
            # Apply crop if in cropped view mode
            if self.cropped_view and self.crop_rect:
                xx, yy, w, h = self.crop_rect
                if xx >= 0 and yy >= 0 and w > 0 and h > 0 and xx+w <= frame.shape[1] and yy+h <= frame.shape[0]:
                    frame = frame[yy:yy+h, xx:xx+w]
            
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = end_time
            
            vis = frame.copy()
            
            # Draw crop rectangle if in crop mode
            if self.crop_mode and self.crop_start and self.crop_end:
                cv2.rectangle(vis, self.crop_start, self.crop_end, (255, 0, 0), 2)
                cv2.putText(vis, "Press ENTER to crop, ESC to cancel", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw tracking selection polygon only during selection, not after initialization
            if not self.crop_mode and len(self.pts) > 0 and self.show_selection:
                for i in range(len(self.pts)):
                    cv2.circle(vis, self.pts[i], 4, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(vis, self.pts[i-1], self.pts[i], (0, 255, 0), 2)
                if len(self.pts) == 4:
                    cv2.line(vis, self.pts[3], self.pts[0], (0, 255, 0), 2)
            
            # Initialize tracking when 4 points are selected
            if len(self.pts) == 4 and not self.initialized:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.array(self.pts, np.int32), 255)
                xx, yy, w, h = cv2.boundingRect(np.array(self.pts, np.int32))
                roi = frame[yy:yy+h, xx:xx+w]
                mask_roi = mask[yy:yy+h, xx:xx+w]
                
                # Extract color histogram
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                self.roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [180, 256], [0, 180, 0, 256])
                cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
                
                # Store object features for re-identification
                self.object_features = self.store_object_features(frame, (xx, yy, w, h))
                
                # Initialize Kalman filter
                self.kf = self.init_kalman()
                self.kf.statePost[:2,0] = np.array([xx + w/2, yy + h/2], np.float32)
                self.initialized = True
                self.track_window = (xx, yy, w, h)
                # Hide selection points after initialization
                self.show_selection = False
                logger.client_logger.debug("Multi-feature tracking self.initialized.")
            
            # Tracking loop
            if self.initialized and not self.crop_mode:
                # Kalman prediction
                pred = self.kf.predict()
                
                # Detect object - use full frame search when object is lost
                search_entire_frame = self.object_lost and (self.frame_index - self.last_seen_frame) < self.max_lost_time
                detection = self.detect_object(frame, self.roi_hist, gabor_filters, search_entire_frame)
                
                if detection is not None:
                    xx, yy, w, h = detection
                    cx = xx + w/2
                    cy = yy + h/2
                    
                    # Update Kalman with measurement
                    meas = np.array([[cx], [cy]], np.float32)
                    self.kf.correct(meas)
                    
                    # Update tracking window
                    self.track_window = detection
                    self.object_lost = False
                    self.lost_counter = 0
                    self.last_seen_frame = self.frame_index
                    
                    # Update object features periodically for better re-identification
                    if self.frame_index % 30 == 0:
                        new_features = self.store_object_features(frame, detection)
                        if new_features is not None:
                            self.object_features = new_features
                    
                    # Draw detection rectangle
                    cv2.rectangle(vis, (xx, yy), (xx+w, yy+h), (0, 255, 0), 2)
                    
                    # Print coordinates
                    # logger.client_logger.debug(f"Object coordinates: x={xx}, y={yy}, w={w}, h={h}, center=({cx:.1f}, {cy:.1f})")

                    self.mallet.pos.x, self.mallet.pos.y = cx/720*self.__screen_width, cy/1280*self.__screen_height
                    
                    # Display coordinates on frame
                    cv2.putText(vis, f"Pos: ({xx}, {yy})", (xx, yy-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(vis, f"Size: {w}x{h}", (xx, yy+h+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Object lost, increment counter
                    self.lost_counter += 1
                    
                    # Check if object is lost for too long
                    if self.lost_counter > self.lost_threshold:
                        self.object_lost = True
                        
                    # If object lost but within max lost time, keep predicting with Kalman
                    if self.object_lost and (self.frame_index - self.last_seen_frame) < self.max_lost_time:
                        # Use Kalman prediction to estimate position
                        x_pred = int(pred[0])
                        y_pred = int(pred[1])
                        
                        # Draw predicted position with different color
                        if self.track_window:
                            xx, yy, w, h = self.track_window
                            pred_rect = (max(0, x_pred-w//2), max(0, y_pred-h//2), w, h)
                            cv2.rectangle(vis, (pred_rect[0], pred_rect[1]), 
                                         (pred_rect[0]+pred_rect[2], pred_rect[1]+pred_rect[3]), 
                                         (0, 165, 255), 2)  # Orange for prediction
                
                # Use filtered state for drawing the predicted position
                x_f, y_f = int(pred[0]), int(pred[1])
                cv2.circle(vis, (x_f, y_f), 8, (0, 0, 255), -1)
                
                # Display status
                if self.object_lost:
                    cv2.putText(vis, "OBJECT LOST", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 0, 255), 2)
                    
                    # Show frames since last detection
                    frames_lost = self.frame_index - self.last_seen_frame
                    cv2.putText(vis, f"Frames since last seen: {frames_lost}", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                else:
                    cv2.putText(vis, "TRACKING", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
            
            # Display mode and FPS
            if self.crop_mode:
                cv2.putText(vis, "CROP MODE", (vis.shape[1]-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif self.cropped_view:
                cv2.putText(vis, "CROPPED VIEW", (vis.shape[1]-180, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.putText(vis, f"FPS: {fps:.1f}", (vis.shape[1]-120, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display help text
            cv2.putText(vis, "Press 'c' for crop mode, 'r' to reset", (10, vis.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Multi-Feature Tracker", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                if self.crop_mode:
                    # Cancel crop mode
                    self.crop_mode = False
                    self.crop_start = None
                    self.crop_end = None
                else:
                    break
            elif key == ord('r'):
                # Reset everything
                self.pts.clear()
                self.selection_complete = False
                self.initialized = False
                self.kf = None
                self.roi_hist = None
                self.track_window = None
                self.object_lost = False
                self.lost_counter = 0
                self.crop_mode = False
                self.cropping = False
                self.crop_start = None
                self.crop_end = None
                self.cropped_view = False
                self.crop_rect = None
                self.object_features = None
                self.show_selection = True  # Re-enable selection display for next selection
                logger.client_logger.debug("Reset all.")
            elif key == ord('c'):
                # Toggle crop mode
                self.crop_mode = not self.crop_mode
                if self.crop_mode:
                    logger.client_logger.debug("Crop mode activated. Draw rectangle to crop.")
                    self.crop_start = None
                    self.crop_end = None
                else:
                    logger.client_logger.debug("Crop mode deactivated.")
            elif key == 13:  # ENTER key
                if self.crop_mode and self.crop_rect:
                    self.cropped_view = not self.cropped_view
                    if self.cropped_view:
                        logger.client_logger.debug(f"Switched to cropped view: {self.crop_rect}")
                    else:
                        logger.client_logger.debug("Switched to full view")
                    self.crop_mode = False

            self.screen.fill((14, 14, 14))

            pygame.draw.line(self.screen, (255, 255, 255), (0, self.__screen_height//2),
                             (self.__screen_width, self.__screen_height//2))

            # mx, my = pygame.mouse.get_pos()
            self.mallet.pos.x = pygame.math.clamp(self.mallet.pos.x, self.mallet.radius,
                                                  self.__screen_width - self.mallet.radius)
            self.mallet.pos.y = clamp(self.mallet.pos.y, self.__screen_height//2 + self.mallet.radius,
                                      self.__screen_height - self.mallet.radius)

            self.mallet.draw(self.screen)

            # get mallet velocity
            self.mallet.update_velocity(self.mallet.pos.x, self.mallet.pos.y)
            # logger.client_logger.debug(self.mallet.velocity)

            # get the position of the puck from the server (physics will be calculated on server)
            ITERATIONS = 100
            for _ in range(ITERATIONS):
                self.physics()
                self.puck.pos.x += self.puck.velocity.x/ITERATIONS
                self.puck.pos.y += self.puck.velocity.y/ITERATIONS

                if (self.puck.pos.x < self.puck.radius or self.puck.pos.x > self.__screen_width - self.puck.radius):
                    self.puck.velocity.x *= -1
                if (self.puck.pos.y < self.puck.radius or self.puck.pos.y > self.__screen_height - self.puck.radius):
                    self.puck.velocity.y *= -1

            # self.puck.velocity *= 0.995

            # draw the puck
            self.puck.draw(self.screen)

            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    self.running = False

            self.delta_time = self.clock.tick(60) / 1000
            self.delta_time = clamp(self.delta_time, 0.001, 0.1)
            pygame.display.flip()

    # Optimize Gabor filters - fewer parameters for speed
    def build_gabor_filters(self):
        filters = []
        ksize = 11  # Even smaller kernel for better speed
        # Reduced parameter combinations
        for theta in [0, np.pi/2]:  # Just 2 orientations
            for sigma in [1.5]:     # Single scale
                for lambd in [np.pi/4]:  # Single wavelength
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                    kernel /= 1.5*kernel.sum()
                    filters.append(kernel)
        return filters
    
    # Optimized texture feature extraction
    def extract_texture_features(self, image, filters):
        # Downscale for speed
        small_img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        features = []
        for kernel in filters:
            filtered = cv2.filter2D(small_img, cv2.CV_8UC3, kernel)
            features.append(filtered)
        # Upscale back to original size
        return cv2.resize(np.mean(features, axis=0), (image.shape[1], image.shape[0]))
    
    # Faster shape detection with reduced processing
    def detect_shapes(self, gray_image):
        # Use lower resolution for edge detection
        small_gray = cv2.resize(gray_image, (0,0), fx=0.5, fy=0.5)
        edges = cv2.Canny(small_gray, 50, 150)
        # Upscale back to original size
        return {'edges': cv2.resize(edges, (gray_image.shape[1], gray_image.shape[0]))}
    
    # Optimized object detection with re-identification capability
    def detect_object(self, frame, roi_hist, gabor_filters, search_entire_frame=False):
        # Convert to HSV and calculate back projection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        back_proj = cv2.filter2D(back_proj, -1, kernel)
        
        # Extract texture features (only if color detection is weak)
        max_back_proj = np.max(back_proj)
        if max_back_proj < 100:  # Only use texture when color detection is weak
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            texture_response = self.extract_texture_features(gray, gabor_filters)
            texture_response = cv2.normalize(texture_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Extract edge features
            shapes = self.detect_shapes(gray)
            edge_density = shapes['edges']
            
            # Weighted combination
            combined_map = cv2.addWeighted(back_proj, 0.7, texture_response, 0.2, 0)
            combined_map = cv2.addWeighted(combined_map, 0.9, edge_density, 0.1, 0)
        else:
            # Just use back projection if it's strong enough
            combined_map = back_proj
        
        # Threshold and find contours
        _, thresh = cv2.threshold(combined_map, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area (faster than checking circularity for all)
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 150:  # Minimum area threshold
                continue
            valid_contours.append(c)
        
        if not valid_contours:
            return None
        
        # Get largest contour as best match
        best_contour = max(valid_contours, key=cv2.contourArea)
        return cv2.boundingRect(best_contour)
    
    # --- Mouse callback for tracking and self.cropping ---
    def on_mouse(self, event, x, y, flags, param):
        # global self.pts, self.selection_complete, self.crop_mode, self.cropping, self.crop_start, self.crop_end, self.cropped_view, self.crop_rect
        
        # Handle self.cropping mode
        if self.crop_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.cropping = True
                self.crop_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
                self.crop_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.cropping = False
                self.crop_end = (x, y)
                # Calculate rectangle coordinates
                x_start = min(self.crop_start[0], self.crop_end[0])
                y_start = min(self.crop_start[1], self.crop_end[1])
                width = abs(self.crop_start[0] - self.crop_end[0])
                height = abs(self.crop_start[1] - self.crop_end[1])
                self.crop_rect = (x_start, y_start, width, height)
                logger.client_logger.debug(f"Crop rectangle: {self.crop_rect}")
        # Handle tracking point selection
        elif not self.crop_mode and len(self.pts) < 4 and event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))
            logger.client_logger.debug(f"Point {len(self.pts)}: {(x, y)}")
            if len(self.pts) == 4:
                self.selection_complete = True
                logger.client_logger.debug("Polygon defined. Initializing tracking...")
    
    # --- Kalman filter init ---
    def init_kalman(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        # Increased process noise for better adaptation to motion changes
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 2e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        return self.kf
    
    # Store object appearance for re-identification
    def store_object_features(self, frame, bbox):
        x, y, w, h = bbox
        if x < 0 or y < 0 or x+w >= frame.shape[1] or y+h >= frame.shape[0]:
            return None
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
            
        # Convert to HSV and calculate histogram
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def physics(self) -> None:
        d = glm.distance(self.mallet.pos, self.puck.pos)
        if (d < self.mallet.radius + self.puck.radius):
            # logger.client_logger.info("Collision!")
            pygame.mixer.music.play()

            # Calculate velocity for puck
            m1 = self.puck.mass
            m2 = self.mallet.mass
            v1 = self.puck.velocity
            v2 = self.mallet.velocity
            x1 = self.puck.pos
            x2 = self.mallet.pos

            overlap = d - (self.mallet.radius + self.puck.radius)
            dir = (x2 - x1) * (overlap * 0.5) / glm.length(x2 - x1)
            self.puck.pos += dir
            self.puck.pos.x = clamp(
                self.puck.pos.x, self.puck.radius, self.__screen_width - self.puck.radius)
            self.puck.pos.y = clamp(
                self.puck.pos.y, self.puck.radius, self.__screen_height - self.puck.radius)

            self.puck.velocity += ((2*m2)/(m1 + m2)) * (glm.dot(v2 - v1,
                                                                x2 - x1) / glm.length2((x2 - x1))) * (x2 - x1)

            self.puck.velocity.x = clamp(self.puck.velocity.x, -30, 30)
            self.puck.velocity.y = clamp(self.puck.velocity.y, -30, 30)
