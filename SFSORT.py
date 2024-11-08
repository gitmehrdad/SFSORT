# ******************************************************************** #
# ****************** Sharif University of Technology ***************** #
# *************** Department of Electrical Engineering *************** #
# ************************ Deep Learning Lab ************************* #
# ************************ SFSORT Version 4.2 ************************ #
# ************ Authors: Mehrdad Morsali - Zeinab Sharifi ************* #
# *********** mehrdadmorsali@gmail.com - zsh.5ooo@gmail.com ********** #
# ******************************************************************** #


# ******************************************************************** #
# ********************** Packages and Libraries ********************** #
# ******************************************************************** #
import numpy as np


use_lap=True
try:
    import lap
except ImportError:
    from scipy.optimize import linear_sum_assignment
    use_lap=False
 
# ******************************************************************** #
# ***************************** Classes ****************************** #
# ******************************************************************** # 
class DotAccess(dict):
    """Provides dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
class TrackState:
    """Enumeration of possible states of a track"""    
    Active = 0
    Lost_Central = 1
    Lost_Marginal = 2
 

class Track:
    """Handles basic track attributes and operations"""
    
    def __init__(self, bbox, frame_id, track_id):
        """Track initialization"""                 
        self.track_id = track_id
        self.bbox = bbox 
        self.state = TrackState.Active                      
        self.last_frame = frame_id
 
    def update(self, box, frame_id):
        """Updates a matched track"""      
        self.bbox = box
        self.state = TrackState.Active
        self.last_frame = frame_id
    

class SFSORT:
    """Multi-Object Tracking System"""
    
    def __init__(self, args):
        """Initialize a tracker with given arguments""" 
        args = DotAccess(args)     

        # Register tracking arguments, setting default values if the argument is not provided      
        if args.high_th is None:
            self.high_th = 0.6
        else:
            self.high_th = self.clamp(args.high_th, 0, 1)
            
        if args.match_th_first is None:
            self.match_th_first = 0.67
        else:
            self.match_th_first = self.clamp(args.match_th_first, 0, 0.67)  
            
        if args.new_track_th is None:
            self.new_track_th = 0.7
        else:
            self.new_track_th = self.clamp(args.new_track_th, self.high_th, 1)          
        
        if args.low_th is None:
            self.low_th = 0.1
        else:
            self.low_th = self.clamp(args.low_th, 0, self.high_th)       
  
        if args.match_th_second is None:
            self.match_th_second = 0.3
        else:
            self.match_th_second = self.clamp(args.match_th_second, 0, 1)     
  
        self.dynamic_tuning = False
        if args.dynamic_tuning is not None:
            self.cth = 0.5
            self.high_th_m = 0.0
            self.new_track_th_m = 0.0
            self.match_th_first_m = 0.0
            if args.dynamic_tuning:
                self.dynamic_tuning = True
                if args.cth is not None:
                    self.cth = self.clamp(args.cth, args.low_th, 1)      
                if args.high_th_m is not None:
                    self.high_th_m = self.clamp(args.high_th_m, 0.02, 0.1)   
                if args.new_track_th_m is not None:
                    self.new_track_th_m = self.clamp(args.new_track_th_m, 0.02, 0.08)                    
                if args.match_th_first_m is not None:
                    self.match_th_first_m = self.clamp(args.match_th_first_m, 0.02, 0.08)   
                    
        if args.marginal_timeout is None:
            self.marginal_timeout = 0
        else:
            self.marginal_timeout = self.clamp(args.marginal_timeout, 0, 500) 
            
        if args.central_timeout is None:
            self.central_timeout = 0
        else:
            self.central_timeout = self.clamp(args.central_timeout, 0, 1000)            
        
        self.l_margin = 0
        self.r_margin = 0
        if args.frame_width:
            self.r_margin = args.frame_width
            if args.horizontal_margin is not None:                             
                self.l_margin = self.clamp(args.horizontal_margin, 0, args.frame_width) 
                self.r_margin = self.clamp(args.frame_width - args.horizontal_margin, 0, args.frame_width) 
        
        self.t_margin = 0
        self.b_margin = 0
        if args.frame_height:
            self.b_margin = args.frame_height
            if args.vertical_margin is not None:                             
                self.t_margin = self.clamp(args.vertical_margin, 0, args.frame_height) 
                self.b_margin = self.clamp(args.frame_height - args.vertical_margin , 0, args.frame_height)   
               
        # Initialize the tracker
        self.frame_no = 0      
        self.id_counter = 0       
        self.active_tracks = []         
        self.lost_tracks = [] 
        
    def update(self, boxes, scores):
        """Updates tracker with new detections"""
        # Adjust dynamic arguments
        hth = self.high_th 
        nth = self.new_track_th 
        mth = self.match_th_first     
        if self.dynamic_tuning:
            count = len(scores[scores>self.cth])     
            if count < 1:
              count = 1
              
            lnc = np.log10(count)           
            hth = self.clamp(hth - (self.high_th_m * lnc), 0, 1)
            nth = self.clamp(nth + (self.new_track_th_m * lnc), hth, 1)    
            mth = self.clamp(mth - (self.match_th_first_m * lnc), 0, 0.67)  
                  
        # Increase frame number
        self.frame_no += 1
        
        # Variable: Active tracks in the next frame
        next_active_tracks = []
        
        # Remove long-time lost tracks      
        all_lost_tracks = self.lost_tracks.copy()
        for track in all_lost_tracks:
            if track.state == TrackState.Lost_Central:
                if self.frame_no - track.last_frame > self.central_timeout:
                    self.lost_tracks.remove(track)                   
            else:
                if self.frame_no - track.last_frame > self.marginal_timeout:
                    self.lost_tracks.remove(track)     
                    
        # Gather out all previous tracks
        track_pool = self.active_tracks + self.lost_tracks  
        
        # Try to associate tracks with high score detections
        unmatched_tracks = np.array([])
        high_score = scores > hth 
        if high_score.any():
            definite_boxes = boxes[high_score]
            definite_scores = scores[high_score]  
            if track_pool:           
                cost = self.calculate_cost(track_pool, definite_boxes) 
                matches, unmatched_tracks, unmatched_detections = self.linear_assignment(cost, mth) 
                # Update/Activate matched tracks
                for track_idx, detection_idx in matches:
                    box = definite_boxes[detection_idx]
                    track = track_pool[track_idx]                   
                    track.update(box, self.frame_no)
                    next_active_tracks.append(track)
                    # Remove re-identified tracks from lost list
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)
                # Identify eligible unmatched detections as new tracks
                for detection_idx in unmatched_detections:                   
                    if definite_scores[detection_idx] > nth:
                        box = definite_boxes[detection_idx]
                        track = Track(box, self.frame_no, self.id_counter)
                        next_active_tracks.append(track)
                        self.id_counter += 1                   
            else:
            	# Associate tracks of the first frame after object-free/null frames
                for detection_idx, score in enumerate(definite_scores):                   
                    if score > nth:
                        box = definite_boxes[detection_idx]
                        track = Track(box, self.frame_no, self.id_counter)
                        next_active_tracks.append(track)
                        self.id_counter += 1  
        
        # Add unmatched tracks to the lost list
        unmatched_track_pool = []                                         
        for track_address in unmatched_tracks:
            unmatched_track_pool.append(track_pool[track_address])   
        next_lost_tracks = unmatched_track_pool.copy()
        
        # Try to associate remained tracks with intermediate score detections 
        intermediate_score = np.logical_and((self.low_th < scores), (scores < hth))      
        if intermediate_score.any(): 
            if len(unmatched_tracks):                 
                possible_boxes = boxes[intermediate_score]
                cost = self.calculate_cost(unmatched_track_pool, possible_boxes, iou_only=True)
                matches, unmatched_tracks, unmatched_detections = self.linear_assignment(cost, self.match_th_second)
                # Update/Activate matched tracks
                for track_idx, detection_idx in matches:
                    box = possible_boxes[detection_idx]
                    track = unmatched_track_pool[track_idx]                  
                    track.update(box, self.frame_no)
                    next_active_tracks.append(track)
                    # Remove re-identified tracks from lost list 
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)
                    next_lost_tracks.remove(track)                                  
            
        # All tracks are lost if there are no detections!
        if not (high_score.any() or  intermediate_score.any()):
            next_lost_tracks = track_pool.copy()
            
        # Update the list of lost tracks
        for track in next_lost_tracks:
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)
                u = track.bbox[0] + (track.bbox[2] - track.bbox[0])/2             
                v = track.bbox[1] + (track.bbox[3] - track.bbox[1])/2
                if (self.l_margin < u < self.r_margin) and (self.t_margin < v < self.b_margin):                   
                    track.state = TrackState.Lost_Central                        
                else:
                    track.state = TrackState.Lost_Marginal                      
                           
        # Update the list of active tracks
        self.active_tracks = next_active_tracks.copy()

        return np.asarray([[x.bbox, x.track_id] for x in next_active_tracks], dtype=object)

    @staticmethod
    def clamp(value, min_value, max_value):
        """ Clamps a value within the specified minimum and maximum bounds."""
        return max(min_value, min(value, max_value))

    @staticmethod
    def calculate_cost(tracks, boxes, iou_only=False):
        """Calculates the association cost based on IoU and box similarity"""
        eps = 1e-7
        active_boxes = [track.bbox for track in tracks]   
           
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = np.array(active_boxes).T
        b2_x1, b2_y1, b2_x2, b2_y2 = np.array(boxes).T      
        
        h_intersection = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0)
        w_intersection = (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)
        
        # Calculate the intersection area
        intersection =  h_intersection * w_intersection
                     
        # Calculate the union area
        box1_height = b1_x2 - b1_x1
        box2_height = b2_x2 - b2_x1
        box1_width = b1_y2 - b1_y1   
        box2_width = b2_y2 - b2_y1

        box1_area = box1_height * box1_width
        box2_area = box2_height * box2_width
        
        union = (box2_area + box1_area[:, None] - intersection + eps)
        
        # Calculate the IoU 
        iou = intersection / union
        
        if iou_only:
            return 1.0 - iou
        
        # Calculate the DIoU                    
        centerx1 = (b1_x1 + b1_x2) / 2.0
        centery1 = (b1_y1 + b1_y2) / 2.0
        centerx2 = (b2_x1 + b2_x2) / 2.0
        centery2 = (b2_y1 + b2_y2) / 2.0        
        inner_diag = np.abs(centerx1[:, None] - centerx2) + np.abs(centery1[:, None] - centery2)
           
        xxc1 = np.minimum(b1_x1[:, None], b2_x1)
        yyc1 = np.minimum(b1_y1[:, None], b2_y1)
        xxc2 = np.maximum(b1_x2[:, None], b2_x2)
        yyc2 = np.maximum(b1_y2[:, None], b2_y2)              
        outer_diag = np.abs(xxc2 - xxc1) + np.abs(yyc2 - yyc1)
       
        diou = iou - (inner_diag / outer_diag)
        
        # Calculate the BBSI 
        delta_w = np.abs(box2_width - box1_width[:, None])
        sw = w_intersection / np.abs(w_intersection + delta_w + eps)
        
        delta_h = np.abs(box2_height - box1_height[:, None])     
        sh = h_intersection / np.abs(h_intersection + delta_h + eps)
               
        bbsi = diou + sh + sw    
        
        # Normalize the BBSI
        cost = (bbsi)/3.0

        return 1.0 - cost 
 
    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """Linear assignment"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

        if use_lap:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  
            matches = np.array([[row, col] for row, col in zip(row_ind, col_ind) if cost_matrix[row, col] <= thresh])  
            matched_rows = set(row_ind)  
            matched_cols = set(col_ind)  
            unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_rows])  
            unmatched_b = np.array([j for j in range(cost_matrix.shape[1]) if j not in matched_cols])  
            
        return matches, unmatched_a, unmatched_b
    