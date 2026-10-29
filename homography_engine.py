import cv2
import numpy as np
from ultralytics import YOLO

class HomographyEngine:
    def __init__(self, model_path, device):
        print(f"[ENGINE] Loading Homography model on {device}...")
        self.pitch_model = YOLO(str(model_path), task="pose")
        self.device = device
        
        # Stabilization state
        self.last_valid_H = None
        self.EMA_ALPHA = 0.85
        
        # 🟢 VISUAL CACHE: Stores drawn points to prevent strobe flickering on non-inference frames
        self.cached_visual_points = []
        
        # Dimensions (10px = 1 Yard)
        self.PITCH_WIDTH = 1200  
        self.PITCH_HEIGHT = 533  
        
        H = self.PITCH_HEIGHT
        self.ABSOLUTE_PITCH_MAP = {
            # 10 Yards (Originals)
            0: [600, 0], 1: [600, H], 
            2: [500, 0], 3: [500, H], 4: [700, 0], 5: [700, H],
            6: [400, 0], 7: [400, H], 8: [800, 0], 9: [800, H],
            10: [300, 0], 11: [300, H], 12: [900, 0], 13: [900, H],
            14: [200, 0], 15: [200, H], 16: [1000, 0], 17: [1000, H],
            
            # 5 Yards (New marks)
            18: [550, 0], 19: [550, H], 20: [650, 0], 21: [650, H],
            22: [450, 0], 23: [450, H], 24: [750, 0], 25: [750, H],
            26: [350, 0], 27: [350, H], 28: [850, 0], 29: [850, H],
            30: [250, 0], 31: [250, H], 32: [950, 0], 33: [950, H],
            34: [150, 0], 35: [150, H], 36: [1050, 0], 37: [1050, H],
            
            # Goal Lines (Yard 0)
            38: [100, 0], 39: [100, H], 40: [1100, 0], 41: [1100, H],
            
            # Endzones (Absolute pitch limits)
            42: [0, 0], 43: [0, H], 44: [1200, 0], 45: [1200, H]
        }

        self.KEYPOINT_NAMES = {
            0: "00_Y50_Top", 1: "01_Y50_Bot", 
            2: "02_Y40L_Top", 3: "03_Y40L_Bot", 4: "04_Y40R_Top", 5: "05_Y40R_Bot", 
            6: "06_Y30L_Top", 7: "07_Y30L_Bot", 8: "08_Y30R_Top", 9: "09_Y30R_Bot", 
            10: "10_Y20L_Top", 11: "11_Y20L_Bot", 12: "12_Y20R_Top", 13: "13_Y20R_Bot", 
            14: "14_Y10L_Top", 15: "15_Y10L_Bot", 16: "16_Y10R_Top", 17: "17_Y10R_Bot",
            18: "18_Y45L_Top", 19: "19_Y45L_Bot", 20: "20_Y45R_Top", 21: "21_Y45R_Bot",
            22: "22_Y35L_Top", 23: "23_Y35L_Bot", 24: "24_Y35R_Top", 25: "25_Y35R_Bot",
            26: "26_Y25L_Top", 27: "27_Y25L_Bot", 28: "28_Y25R_Top", 29: "29_Y25R_Bot",
            30: "30_Y15L_Top", 31: "31_Y15L_Bot", 32: "32_Y15R_Top", 33: "33_Y15R_Bot",
            34: "34_Y5L_Top", 35: "35_Y5L_Bot", 36: "36_Y5R_Top", 37: "37_Y5R_Bot",
            38: "38_Y0L_Top", 39: "39_Y0L_Bot", 40: "40_Y0R_Top", 41: "41_Y0R_Bot",
            42: "42_EndzoneL_Top", 43: "43_EndzoneL_Bot", 44: "44_EndzoneR_Top", 45: "45_EndzoneR_Bot"
        }

    def draw_2d_pitch(self):
        """Generates a clean 2D tactical field background."""
        pitch = np.zeros((self.PITCH_HEIGHT, self.PITCH_WIDTH, 3), dtype=np.uint8)
        pitch[:] = (40, 150, 40)
        
        cv2.line(pitch, (100, 0), (100, self.PITCH_HEIGHT), (255, 255, 255), 2)
        cv2.line(pitch, (1100, 0), (1100, self.PITCH_HEIGHT), (255, 255, 255), 2)
        
        for x_coord in range(200, 1100, 100):
            cv2.line(pitch, (x_coord, 0), (x_coord, self.PITCH_HEIGHT), (220, 220, 220), 1)
            
        for x_coord in range(150, 1100, 100):
            cv2.line(pitch, (x_coord, 0), (x_coord, self.PITCH_HEIGHT), (150, 190, 150), 1)
            
        cv2.line(pitch, (600, 0), (600, self.PITCH_HEIGHT), (255, 255, 255), 3)
        return pitch

    def draw_cached_points(self, frame_disp):
        """Redraws cached visual points cleanly on skipped inference frames."""
        if frame_disp is None or not self.cached_visual_points: return
        for pt_data in self.cached_visual_points:
            x, y = pt_data['pt']
            color = pt_data['color']
            radius = pt_data['radius']
            cv2.circle(frame_disp, (x, y), radius, color, -1)
            if 'text' in pt_data and pt_data['text']:
                cv2.putText(frame_disp, pt_data['text'], (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_frame(self, frame, frame_disp=None, run_inference=True):
        """Extracts field keypoints and calculates homography. Uses visual cache on skipped frames."""
        # 🟢 FAST CACHE PATH: Redraw previous visual markers without running GPU inference
        if not run_inference:
            self.draw_cached_points(frame_disp)
            return self.last_valid_H

        results = self.pitch_model.predict(frame, verbose=False, device=self.device, conf=0.557, save=False)
        if not results or len(results[0].keypoints.xy) == 0: 
            self.draw_cached_points(frame_disp)
            return self.last_valid_H
        
        kp = results[0].keypoints.xy[0].cpu().numpy()
        confidences = results[0].keypoints.conf[0].cpu().numpy()
        
        top_candidates, bottom_candidates = [], []
        for i, point in enumerate(kp):
            if i in self.ABSOLUTE_PITCH_MAP and i < len(confidences):
                if confidences[i] > 0.402 and point[0] > 0 and point[1] > 0:
                    if i % 2 == 0: top_candidates.append({'id': i, 'pt': point})
                    else:          bottom_candidates.append({'id': i, 'pt': point})
                        
        if len(top_candidates) == 0 or len(bottom_candidates) == 0: 
            self.draw_cached_points(frame_disp)
            return self.last_valid_H
        
        median_top_y = np.median([c['pt'][1] for c in top_candidates])
        median_bottom_y = np.median([c['pt'][1] for c in bottom_candidates])
        y_ecuador = (median_top_y + median_bottom_y) / 2.0
        
        valid_points = []
        new_visual_cache = []
        
        for c in top_candidates:
            if c['pt'][1] < y_ecuador: 
                valid_points.append(c)
            else: 
                new_visual_cache.append({'pt': (int(c['pt'][0]), int(c['pt'][1])), 'radius': 8, 'color': (0, 0, 255)})
                
        for c in bottom_candidates:
            if c['pt'][1] > y_ecuador: 
                valid_points.append(c)
            else: 
                new_visual_cache.append({'pt': (int(c['pt'][0]), int(c['pt'][1])), 'radius': 8, 'color': (0, 0, 255)})
                
        top_count = sum(1 for c in valid_points if c['id'] % 2 == 0)
        bottom_count = sum(1 for c in valid_points if c['id'] % 2 != 0)
        
        if top_count < 2 or bottom_count < 2: 
            self.draw_cached_points(frame_disp)
            return self.last_valid_H
            
        dst_points, src_points = [], []
        for c in valid_points:
            x_int, y_int = int(c['pt'][0]), int(c['pt'][1])
            dst_points.append([c['pt'][0], c['pt'][1]])
            src_points.append(self.ABSOLUTE_PITCH_MAP[c['id']])
            
            kp_name = self.KEYPOINT_NAMES.get(c['id'], f"ID:{c['id']}")
            new_visual_cache.append({'pt': (x_int, y_int), 'radius': 6, 'color': (0, 255, 0), 'text': kp_name})
                
        # Update cache with newly processed points and draw them immediately
        self.cached_visual_points = new_visual_cache
        self.draw_cached_points(frame_disp)
                
        dst_points = np.array(dst_points, dtype=np.float32)
        src_points = np.array(src_points, dtype=np.float32)
        
        new_H, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 10.0)
        
        if new_H is not None:
            if self.last_valid_H is None:
                self.last_valid_H = new_H
            else:
                self.last_valid_H = (self.EMA_ALPHA * self.last_valid_H) + ((1.0 - self.EMA_ALPHA) * new_H)
                
        return self.last_valid_H