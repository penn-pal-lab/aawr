"""
This script creates a color segmentation mask from an input image and can apply it to a video.
The user can click on an object to sample its color and adjust HSV thresholds.
The user can also define a crop region by clicking two points for the top-left and bottom-right corners.
"""

import numpy as np
import cv2
import os
import yaml
import sys
import json

class ColorSegmentation:
    def __init__(self, image_path, video_path=None, output_dir='results_pi0', config_output_path=None):
        self.image_path = image_path
        self.video_path = video_path
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if video_path and not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        self.config_output_path = config_output_path
        os.makedirs(config_output_path, exist_ok=True)
        
        self.clicked_colors = []
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([179, 255, 255])

        self.crop_mode = False
        self.crop_points = []
        self.crop_region = None  # (x1, y1, x2, y2)
        
        self.window_name = "Color Segmentation"
        self.click_count = 0
        self.mouse_pos = (0, 0)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events to sample colors or define crop region"""
        # Track mouse position for drawing the rectangle in crop mode
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get current frame
            frame = param
            if frame is not None:
                if self.crop_mode:
                    # In crop mode, record points for the crop region
                    self.crop_points.append((x, y))
                    print(f"Crop point {len(self.crop_points)}: ({x}, {y})")
                    
                    # If we have two points, define the crop region
                    if len(self.crop_points) == 2:
                        x1 = min(self.crop_points[0][0], self.crop_points[1][0])
                        y1 = min(self.crop_points[0][1], self.crop_points[1][1])
                        x2 = max(self.crop_points[0][0], self.crop_points[1][0])
                        y2 = max(self.crop_points[0][1], self.crop_points[1][1])
                        
                        self.crop_region = (x1, y1, x2, y2)
                        self.crop_mode = False  # Exit crop mode
                        print(f"Crop region defined: {self.crop_region}")
                else:
                    # Normal color sampling mode
                    # Convert BGR to HSV
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    # Sample color at the clicked point
                    color = hsv_frame[y, x]
                    self.clicked_colors.append(color)
                    self.click_count += 1
                    print(f"Sampled color {self.click_count}: HSV = {color}")
                    
                    # Update color ranges based on clicked points
                    if len(self.clicked_colors) > 0:
                        # Calculate min and max for each HSV component
                        hsv_values = np.array(self.clicked_colors)
                        # Add tolerance to the min/max values
                        tolerance_h = 10  # Hue tolerance
                        tolerance_s = 50  # Saturation tolerance
                        tolerance_v = 50  # Value tolerance
                        
                        # Calculate min/max with tolerance
                        h_min = max(0, np.min(hsv_values[:, 0]) - tolerance_h)
                        s_min = max(0, np.min(hsv_values[:, 1]) - tolerance_s)
                        v_min = max(0, np.min(hsv_values[:, 2]) - tolerance_v)
                        
                        h_max = min(179, np.max(hsv_values[:, 0]) + tolerance_h)
                        s_max = min(255, np.max(hsv_values[:, 1]) + tolerance_s)
                        v_max = min(255, np.max(hsv_values[:, 2]) + tolerance_v)
                        
                        # Update trackbar positions
                        cv2.setTrackbarPos('HMin', self.window_name, int(h_min))
                        cv2.setTrackbarPos('SMin', self.window_name, int(s_min))
                        cv2.setTrackbarPos('VMin', self.window_name, int(v_min))
                        cv2.setTrackbarPos('HMax', self.window_name, int(h_max))
                        cv2.setTrackbarPos('SMax', self.window_name, int(s_max))
                        cv2.setTrackbarPos('VMax', self.window_name, int(v_max))

    def create_trackbars(self):
        """Create trackbars for adjusting HSV thresholds"""
        def nothing(x):
            pass
        
        # Create trackbars
        cv2.createTrackbar('HMin', self.window_name, 0, 179, nothing)
        cv2.createTrackbar('SMin', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('VMin', self.window_name, 0, 255, nothing)
        cv2.createTrackbar('HMax', self.window_name, 179, 179, nothing)
        cv2.createTrackbar('SMax', self.window_name, 255, 255, nothing)
        cv2.createTrackbar('VMax', self.window_name, 255, 255, nothing)
    
    def get_threshold_values(self):
        """Get current threshold values from trackbars"""
        h_min = cv2.getTrackbarPos('HMin', self.window_name)
        s_min = cv2.getTrackbarPos('SMin', self.window_name)
        v_min = cv2.getTrackbarPos('VMin', self.window_name)
        h_max = cv2.getTrackbarPos('HMax', self.window_name)
        s_max = cv2.getTrackbarPos('SMax', self.window_name)
        v_max = cv2.getTrackbarPos('VMax', self.window_name)
        
        self.hsv_lower = np.array([h_min, s_min, v_min])
        self.hsv_upper = np.array([h_max, s_max, v_max])
        
        return self.hsv_lower, self.hsv_upper
    
    def apply_mask(self, frame):
        """Apply color segmentation mask to the frame"""
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current threshold values
        lower, upper = self.get_threshold_values()
        
        # Create mask
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Apply mask
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        return mask, result
    
    def apply_mask_without_window(self, yaml_path, frame, cxcy, target_side_length, visualize=False):
        """Apply color segmentation mask to the frame using values from YAML config."""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        try:
            with open(yaml_path, 'r') as f:
                hsv_config = yaml.safe_load(f)
            lower = np.array(hsv_config['lower'], dtype=np.uint8)
            upper = np.array(hsv_config['upper'], dtype=np.uint8)
        except Exception as e:
            raise e

        # Apply the mask using the loaded values
        mask = cv2.inRange(hsv_frame, lower, upper)

        # currently the result is black for pixels that are not in the mask
        if visualize:
            cxcy = np.array(cxcy)
            result = frame
            result[mask == 0] //= 5
            # draw a bounding box on the result where the center is cxcy and the side length is target_side_length
            x1, y1 = cxcy - target_side_length // 2
            x2, y2 = cxcy + target_side_length // 2
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            result = cv2.bitwise_and(frame, frame, mask=mask)
        return mask, result
    
    def apply_crop(self, frame, mask=None, result=None):
        """Apply crop to frame and optionally to mask and result"""
        if self.crop_region is None:
            return frame, mask, result
        
        x1, y1, x2, y2 = self.crop_region
        cropped_frame = frame[y1:y2, x1:x2].copy()
        
        if mask is not None:
            cropped_mask = mask[y1:y2, x1:x2].copy()
        else:
            cropped_mask = None
            
        if result is not None:
            cropped_result = result[y1:y2, x1:x2].copy()
        else:
            cropped_result = None
            
        return cropped_frame, cropped_mask, cropped_result
    
    def draw_crop_region(self, frame):
        """Draw the crop region or crop points on the frame"""
        vis_frame = frame.copy()
        
        # Draw crop region if it exists
        if self.crop_region is not None:
            x1, y1, x2, y2 = self.crop_region
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw crop points if in crop mode
        if self.crop_mode and len(self.crop_points) > 0:
            for i, point in enumerate(self.crop_points):
                cv2.circle(vis_frame, point, 5, (0, 0, 255), -1)
                cv2.putText(vis_frame, f"P{i+1}", (point[0]+10, point[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If one point is selected, draw a rectangle from that point to current mouse position
            if len(self.crop_points) == 1:
                cv2.rectangle(vis_frame, self.crop_points[0], self.mouse_pos, (0, 0, 255), 2)
        
        return vis_frame
    
    def process_video(self):
        """Process the video with current color segmentation and crop settings"""
        if not self.video_path:
            print("No video path provided. Skipping video processing.")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        output_path = os.path.join(self.output_dir, 'processed_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # If crop region exists, adjust output dimensions
        if self.crop_region:
            x1, y1, x2, y2 = self.crop_region
            out_width, out_height = x2-x1, y2-y1
        else:
            out_width, out_height = width, height
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Process each frame
        print(f"Processing video with {frame_count} frames...")
        progress_interval = max(1, frame_count // 100)  # Update progress every 1%
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply mask
            mask, result = self.apply_mask(frame)
            
            # Apply crop if needed
            if self.crop_region:
                _, _, processed_frame = self.apply_crop(frame, mask, result)
            else:
                processed_frame = result
                
            # Write to output video
            out.write(processed_frame)
            
            # Show progress
            frame_idx += 1
            if frame_idx % progress_interval == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{frame_count})")
        
        # Release resources
        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_path}")
    
    def run(self):
        """Run the color segmentation tool with the static image"""
        print("Starting color segmentation tool...")
        print("Click on objects to sample colors")
        print("Adjust sliders to fine-tune the color range")
        print("Press 'C' to define crop region (requires 2 clicks for corners)")
        print("Press 'S' to save current configuration")
        print("Press 'V' to process video (if provided)")
        print("Press 'R' to reset")
        print("Press 'ESC' to exit")
        
        # Create window and trackbars
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.create_trackbars()
        
        # Set mouse callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback, self.image)
        
        try:
            while True:
                # Use the loaded image instead of camera frames
                rgb_frame = self.image.copy()
                
                # Apply mask
                mask, result = self.apply_mask(rgb_frame)
                
                # Draw crop region
                vis_rgb = self.draw_crop_region(rgb_frame)
                vis_result = self.draw_crop_region(result)
                
                # Show cropped version if crop region exists
                if self.crop_region is not None and not self.crop_mode:
                    # Apply crop
                    cropped_frame, cropped_mask, cropped_result = self.apply_crop(rgb_frame, mask, result)
                    
                    # Create small visualization of cropped view
                    # Add a small inset of the cropped region
                    h, w = cropped_frame.shape[:2]
                    max_inset_size = 200
                    scale = min(max_inset_size / w, max_inset_size / h)
                    inset_w, inset_h = int(w * scale), int(h * scale)
                    
                    cropped_resized = cv2.resize(cropped_frame, (inset_w, inset_h))
                    cropped_result_resized = cv2.resize(cropped_result, (inset_w, inset_h))
                    
                    # Create a border around the inset
                    cropped_resized = cv2.copyMakeBorder(cropped_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                    cropped_result_resized = cv2.copyMakeBorder(cropped_result_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                    
                    # Overlay the insets at the bottom right
                    h_main, w_main = vis_rgb.shape[:2]
                    h_inset, w_inset = cropped_resized.shape[:2]
                    
                    # Position the insets
                    x_offset = w_main - w_inset - 10
                    y_offset = h_main - h_inset - 10
                    
                    # Add the insets
                    vis_rgb[y_offset:y_offset+h_inset, x_offset:x_offset+w_inset] = cropped_resized
                    vis_result[y_offset:y_offset+h_inset, x_offset:x_offset+w_inset] = cropped_result_resized
                
                vis_image = np.hstack([vis_rgb, vis_result])
                
                text = f"HSV Range: [{self.hsv_lower[0]},{self.hsv_lower[1]},{self.hsv_lower[2]}] - [{self.hsv_upper[0]},{self.hsv_upper[1]},{self.hsv_upper[2]}]"
                cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_image, f"Clicked points: {self.click_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                crop_status = "CROP MODE ACTIVE - Click two points" if self.crop_mode else ""
                if self.crop_region is not None and not self.crop_mode:
                    x1, y1, x2, y2 = self.crop_region
                    crop_status = f"Crop region: ({x1},{y1}) to ({x2},{y2})"
                
                cv2.putText(vis_image, crop_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, vis_image)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # Save configuration
                    hsv_ranges = {
                        'lower': self.hsv_lower.tolist(),
                        'upper': self.hsv_upper.tolist()
                    }
                    
                    if self.crop_region is not None:
                        hsv_ranges['crop_region'] = self.crop_region
                    
                    yaml_path = os.path.join(self.config_output_path, 'color_segmentation_config.yaml')
                    with open(yaml_path, 'w') as f:
                        yaml.dump(hsv_ranges, f, default_flow_style=False)
                    print(f"Saved configuration to {yaml_path}")
                    np.save(os.path.join(self.config_output_path, 'hsv_ranges.npy'), hsv_ranges)
                    
                    cv2.imwrite(os.path.join(self.config_output_path, 'original.jpg'), rgb_frame)
                    cv2.imwrite(os.path.join(self.config_output_path, 'mask.jpg'), mask)
                    cv2.imwrite(os.path.join(self.config_output_path, 'result.jpg'), result)
                    
                    # Save cropped images if crop region exists
                    if self.crop_region is not None:
                        cropped_frame, cropped_mask, cropped_result = self.apply_crop(rgb_frame, mask, result)
                        cv2.imwrite(os.path.join(self.config_output_path, 'cropped_original.jpg'), cropped_frame)
                        cv2.imwrite(os.path.join(self.config_output_path, 'cropped_mask.jpg'), cropped_mask)
                        cv2.imwrite(os.path.join(self.config_output_path, 'cropped_result.jpg'), cropped_result)
                    print(f"Saved images to {self.config_output_path}")

                elif key == ord('v'):  # Process video if provided
                    if self.video_path:
                        print("Processing video...")
                        self.process_video()
                    else:
                        print("No video path provided. Skipping video processing.")
                elif key == ord('r'):  # Reset
                    self.clicked_colors = []
                    self.click_count = 0
                    self.crop_mode = False
                    self.crop_points = []
                    self.crop_region = None
                    cv2.setTrackbarPos('HMin', self.window_name, 0)
                    cv2.setTrackbarPos('SMin', self.window_name, 0)
                    cv2.setTrackbarPos('VMin', self.window_name, 0)
                    cv2.setTrackbarPos('HMax', self.window_name, 179)
                    cv2.setTrackbarPos('SMax', self.window_name, 255)
                    cv2.setTrackbarPos('VMax', self.window_name, 255)
                    print("Reset color selection and crop region")
                elif key == ord('c'):  # Enter crop mode
                    self.crop_mode = True
                    self.crop_points = []
                    print("Entering crop mode. Click two points to define crop region.")
        finally:    
            cv2.destroyAllWindows()


if __name__ == "__main__":
    TRAJ_ROOT = sys.argv[1]
    example_image_path = sys.argv[2]

    DATA_DIRS = ["success", "failure"]
    OUTPUT_DIR = "color_segmentation"
    NPY_OUTPUT_DIR = "color_segmentation_npy"
    CONFIG_OUTPUT_DIR = "color_seg_config"
    CONFIG_OUTPUT_PATH = os.path.join(TRAJ_ROOT, CONFIG_OUTPUT_DIR)
    YAML_OUTPUT_PATH = os.path.join(CONFIG_OUTPUT_PATH, "color_segmentation_config.yaml")    
    VISUALIZE = True # turn off for speed
    CXCY = (640, 200)
    TARGET_SIDE_LENGTH = 400

    h, w = cv2.imread(example_image_path).shape[:2]
    segmentation = ColorSegmentation(example_image_path, video_path=None, output_dir=CONFIG_OUTPUT_DIR, config_output_path=CONFIG_OUTPUT_PATH)
    segmentation.run()

    for data_dir in DATA_DIRS:
        for traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir)):
            for sub_traj_dir in os.listdir(os.path.join(TRAJ_ROOT, data_dir, traj_dir)):
                white_list_path =  os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir, "white_list.json")
                color_seg_folder = os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir, "color_segmentation")
                
                with open(white_list_path, 'r') as f:
                    white_list = json.load(f)

                print("=======", os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir))
                mask_output_path = os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir, OUTPUT_DIR)
                os.makedirs(mask_output_path, exist_ok=True)
                npy_output_path = os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir, NPY_OUTPUT_DIR)
                os.makedirs(npy_output_path, exist_ok=True)
                image_dir = os.path.join(os.path.join(TRAJ_ROOT, data_dir, traj_dir, sub_traj_dir), "recordings", "frames", "hand_camera")
                masks = np.zeros((len(white_list), h, w), dtype=bool)
                sorted_images = sorted(os.listdir(image_dir))

                for i, idx in enumerate(white_list):
                    image_name = f"{idx:05d}.jpg"
                    raw_image_path = os.path.join(image_dir, image_name)
                    mask, result = segmentation.apply_mask_without_window(YAML_OUTPUT_PATH, cv2.imread(raw_image_path), CXCY, TARGET_SIDE_LENGTH, VISUALIZE)
                    visualization_image_path = os.path.join(mask_output_path, image_name)
                    cv2.imwrite(visualization_image_path, result)
                    masks[i] = mask.astype(bool)

                original_shape = masks.shape
                packed_masks = np.packbits(masks, axis=-1)
                output_npz_path = os.path.join(npy_output_path, "new_masks_packed.npz")
                np.savez_compressed(output_npz_path, packed_masks=packed_masks, original_shape=original_shape, idxs=white_list)

    