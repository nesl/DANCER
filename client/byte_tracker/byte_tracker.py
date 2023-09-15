import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import pdb
import cv2

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

# Checks if we are within the buffer area of the image edge
def close_to_image_edge(buffer, tlbr):

    is_close = False
    # Check if the x position is next to image edge
    if tlbr[0] < buffer:
        is_close = True 

    return is_close



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, detected_class, detected_extra, enter_anywhere=True):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.detected_class = [detected_class]
        self.voted_class = detected_class
        self.detected_extra = [detected_extra]
        self.enter_anywhere = enter_anywhere
        

        self.max_window_detected = 150

        # Keep track of recent window of movement
        self.location_window = []
        # Keep track of cases where we had significant movement
        self.significant_movement_window = []

        

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    # Check if a newly assigned track is allowed
    def allowed_track_entrance(self, bbox):
        allowed = False

        if bbox[0] > 0 and bbox[1] > 0 and bbox[2] < 300 and bbox[3] < 1079:
            allowed = True

        return allowed

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))


        # We only start a new tracklet if it emerges from the left side of the screen
        # print(self.tlbr)
        activated = False
        if self.enter_anywhere or self.allowed_track_entrance(self.tlbr):
            
            self.tracklet_len = 0
            self.state = TrackState.Tracked
            if frame_id == 1:
                self.is_activated = True
            # self.is_activated = True
            self.frame_id = frame_id
            self.start_frame = frame_id
            activated = True

            self.location_window.append(self.tlbr)
        
        return activated

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    #  Track must be moving across 
    def is_moving(self, window_size=10):
        is_moving = False
        # Keep track only of the horizontal points
        difference = abs(self.location_window[0][0] - self.location_window[1][0])
        if difference > 5: # Moved 5 pixels
            is_moving = True

        return is_moving


    def update(self, new_track, frame_id, buffer_ignore, ignore_stationary):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Note - only add a new detected class if the vehicle is moving over the most recent
        #   10 frames (must be a difference in location in all 10 frames)
        self.location_window.append(self.tlbr)


        # If we are too close to the edge of the window, we ignore detections
        if not close_to_image_edge(buffer_ignore, new_track.tlbr):

            # Now, if there is enough data in the movement window, check if it's moving
            if len(self.location_window) > 50 or not ignore_stationary:

                # If we choose to not ignore stationary, then we will always add detections.
                #  But if we ignore stationary cases, then we must check if it's moving.
                if self.is_moving() or not ignore_stationary:

                    self.detected_class.extend(new_track.detected_class)
                    self.detected_extra.extend(new_track.detected_extra)
                    self.significant_movement_window.append(new_track.tlbr)
                    # If the movement window exceeds a certain size, cull it.
                    if len(self.significant_movement_window) > 20:
                        self.significant_movement_window.pop(0)

                    # Also cull the detected class and detected extra if we exceed
                    #  a certain amount
                    if len(self.detected_class) > self.max_window_detected:
                        self.detected_class.pop(0)
                        self.detected_extra.pop(0)

                self.location_window.pop(0)
            else:
                self.detected_class.extend(new_track.detected_class)
                self.detected_extra.extend(new_track.detected_extra)


            # predict the class
            self.voted_class = np.bincount(self.detected_class).argmax()
            
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
            self.state = TrackState.Tracked
            self.is_activated = True

            self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, match_class_filter, recover_lost_track, buffer_zone, \
        ignore_stationary, frame_rate, enter_anywhere):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = args.track_buffer
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # other params
        self.match_class_filter = match_class_filter
        self.recover_lost_track = recover_lost_track
        self.ignore_stationary = ignore_stationary

        self.track_id_to_observe = -1
        self.current_image = None
        self.issue_pause = False

        self.buffer_ignore = buffer_zone # We ignore anything within X pixels of the image edge
        self.buffer_zone = buffer_zone

        # Allow tracks to enter from anywhere
        self.enter_anywhere = enter_anywhere

    # Basically matches a given track with a lost track based on a certain amount of overlap
    # And assuming they are the same class
    def match_with_lost(self, track_to_match, match_class_filter, overlap=0.30):

        # Iterate through every lost track, and 
        # # check how much it overlaps with this track.
        ltrack_index = -1
        best_overlap = 0.0
        best_track = None
        tlbr_of_current_track = None

        if track_to_match.detected_class[0] not in match_class_filter:
            return

        # I guess the first that happens it that it looks through lost tracks
        for l_i, ltrack in enumerate(self.lost_stracks):

            if ltrack.track_id == self.track_id_to_observe:
                print("in matching with lost")

            # Iterate through its previous movement windows and check overlap
            window_to_look_at = [ltrack.tlbr]
            window_to_look_at.extend(ltrack.significant_movement_window)
            for prev_tlbr in window_to_look_at:
                # print("Checking here...")
                # Calculate overlap with the given track
                area_overlap = find_area_overlap(prev_tlbr, track_to_match.tlbr)
                measured_overlap = area_overlap / find_area(track_to_match.tlbr)

                if ltrack.track_id == self.track_id_to_observe and measured_overlap > 0.0:
                    print("Checking overlap")
                    print(measured_overlap)

                    cv2.rectangle(self.current_image, \
                    (int(prev_tlbr[0]), int(prev_tlbr[1])), \
                        (int(prev_tlbr[2]), int(prev_tlbr[3])), (255, 0, 0), 5)

                # Check that overlap is good and class matches
                if measured_overlap > overlap and (ltrack.voted_class == track_to_match.voted_class):
                    if best_overlap < measured_overlap:
                        best_overlap = measured_overlap
                        ltrack_index = l_i
                        tlbr_of_current_track = prev_tlbr
                    else:
                        continue
                           
        # If the ltrack returns nothing, meaning that we never found the object
        #  in the kalman predictions, then we have to assign it to the closest
        #  tracked object.  If the measured distance is beyond the default shortest_distance
        #  then no track will be found.
        if ltrack_index == -1:
            for l_i, ltrack in enumerate(self.lost_stracks):

                window_to_look_at = [ltrack.tlbr]
                window_to_look_at.extend(ltrack.significant_movement_window)
                shortest_distance = 100
                for prev_tlbr in window_to_look_at:
                    # Get distance between each point of the bounding box
                    pdistance = shortest_point_distance(prev_tlbr, track_to_match.tlbr)

                    if ltrack.track_id == self.track_id_to_observe and pdistance < 100:
                        print(pdistance)
                        print("matched by distance")

                    # Find if it's the shortest
                    if pdistance < shortest_distance:
                        shortest_distance = pdistance
                        ltrack_index = l_i
                        tlbr_of_current_track = prev_tlbr

        # So the problem is the following:
        #  A vehicle which leaves ends up as a 'lost track'
        #  Another vehicle in the process of leaving gets assigned that 'lost track'
        # So basicically, we have to patch again if this assigned track
        #   overlaps with an existing track
        
        # Iterate through each known track, and make sure the one that we lost
        #  doesn't match with one that currently exists (avoid overwriting)
        for t_track in self.tracked_stracks:

            area_overlap = find_area_overlap(t_track.tlbr, track_to_match.tlbr)
            measured_overlap = area_overlap / find_area(track_to_match.tlbr)

            if t_track.track_id == self.track_id_to_observe:
                print("Patching at end:")
                print(measured_overlap)

            if t_track.track_id == 3609:
                print("Patching at end2:")
                print(measured_overlap)
                

            if measured_overlap > 0.70:
                ltrack_index = -1  # We ignore this track if it matches.

        # Remember to remove this from lost and add it back to tracked
        if ltrack_index >= 0:

            out_track = self.lost_stracks.pop(ltrack_index)
            out_track.update(track_to_match, self.frame_id, self.buffer_ignore, self.ignore_stationary)

            # This is just for debugging
            if out_track.track_id == self.track_id_to_observe:
                print("Matched")
                cv2.rectangle(self.current_image, \
                    (int(track_to_match.tlbr[0]), int(track_to_match.tlbr[1])), \
                        (int(track_to_match.tlbr[2]), int(track_to_match.tlbr[3])), (0, 255, 0), 1)
                self.issue_pause = True


            # restart its movement window
            out_track.significant_movement_window = [out_track.tlbr]
            self.tracked_stracks.append(out_track)

        



    def update(self, output_results, img_info, img_size, detected_class, image, detected_extra):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        self.current_image = image

        # print(output_results.shape)

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        # print(dets_second)
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        # print(scores_second)

        detected_class_keep = detected_class[remain_inds]
        detected_class_second = detected_class[inds_second]

        detected_extra_keep = detected_extra[remain_inds]
        detected_extra_second = detected_extra[inds_second]


        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, detected_class_keep[z_idx], detected_extra_keep[z_idx], self.enter_anywhere) for
                          z_idx,(tlbr, s) in enumerate(zip(dets, scores_keep))]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                if track.track_id == self.track_id_to_observe:
                    print("Adding to unconfirmed")
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, self.buffer_ignore, self.ignore_stationary)
                activated_starcks.append(track)
            else:
                if track.track_id == self.track_id_to_observe:
                    print("re-activating")
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            # print(len(list(zip(dets_second, scores_second))))
            # print(list(zip(dets_second, scores_second)))
            # detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, detected_class_second[z_idx], detected_extra_second[z_idx]) for
            #               z_idx,(tlbr, s) in enumerate(zip(dets_second, scores_second), self.buffer_zone)]
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, detected_class_second[z_idx], detected_extra_second[z_idx], self.enter_anywhere) for
                          z_idx,(tlbr, s) in enumerate(zip(dets_second, scores_second))]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]

            ### TEMPORARY
            if track.track_id == self.track_id_to_observe:
                print("Attempting match with detection")

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.buffer_ignore, self.ignore_stationary)
                
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:

            

            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        unconfirmed_match_to_lost = []
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:

            ### TEMPORARY
            if track.track_id == self.track_id_to_observe:
                print("Matching tracks via IoU")

            unconfirmed[itracked].update(detections[idet], self.frame_id, self.buffer_ignore, self.ignore_stationary)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:

            ### TEMPORARY
            if track.track_id == self.track_id_to_observe:
                print("In unconfirmed")

            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]

            
        
            ### TEMPORARY
            if track.track_id == self.track_id_to_observe:
                print("In u_detection")

            if track.score < self.det_thresh:
                continue


            # Check if we need to active this track - otherwise it is to be matched
            if track.activate(self.kalman_filter, self.frame_id):
                activated_starcks.append(track)
            else:
                unconfirmed_match_to_lost.append(track)


        """ Step 5: Update state"""
        for track in self.lost_stracks:

            ### TEMPORARY
            if track.track_id == self.track_id_to_observe:
                print(self.frame_id)
                print("In lost...")

            if self.frame_id - track.end_frame > self.max_time_lost:

                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        
        # Check the output stracks if our track id is in it
        for otrack in self.tracked_stracks:
            if otrack.track_id == self.track_id_to_observe:
                print(self.frame_id)
                print("added from activated")
        
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        
        # Check the output stracks if our track id is in it
        for otrack in activated_starcks:
            if otrack.track_id == self.track_id_to_observe:
                print(self.frame_id)
                print("added from activated")

        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        # Check the output stracks if our track id is in it
        for otrack in refind_stracks:
            if otrack.track_id == self.track_id_to_observe:
                print(self.frame_id)
                print("added from refind")

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        
        if self.recover_lost_track:
            # Now, as a last fix, check if we need to match any of our lost tracks
            for check_track in unconfirmed_match_to_lost:
                # Here we apply a patch - if the detected object overlaps with a previously 
                #  known track by a certain amount, then set as that new track.
                self.match_with_lost(check_track, self.match_class_filter)


        
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # Check the output stracks if our track id is in it
        for otrack in output_stracks:
            if otrack.track_id == self.track_id_to_observe:
                print(self.frame_id)
                print("In results")

        # Note - we only update tracks if there is sufficient data
        # output_stracks = [track for track in self.tracked_stracks if len(track.detected_class)>1]

        # If we issue the pause do so but reset it
        issue_pause = False
        if self.issue_pause:
            issue_pause = True
            self.issue_pause = False
            # print(output_stracks)

        return output_stracks, issue_pause


def shortest_point_distance(box1, box2):

    # Get all 4 points of the bounding box for each
    points1 = [[box1[0], box1[1]], [box1[0], box1[3]], [box1[2], box1[1]], [box1[2], box1[3]]]
    points2 = [[box2[0], box2[1]], [box2[0], box2[3]], [box2[2], box2[1]], [box2[2], box2[3]]]

    # Get the shortest distance across all points
    shortest_distance = 100
    for p1 in points1:
        for p2 in points2:
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if distance < shortest_distance:
                shortest_distance = distance
    return shortest_distance


def find_area(a):

    width = a[2] - a[0]
    height = a[3] - a[1]
    return width * height


def find_area_overlap(a,b):
    overlap = 0.0
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        overlap = dx*dy
    return overlap

# One important change
#  We avoid significant overlap here since we have an aerial view
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []

    tlista_tlbrs = []

    for t in tlista:  # tlista is usually the tracked_stracks
        exists[t.track_id] = 1
        tlista_tlbrs.append(t.tlbr)
        res.append(t)
    for t in tlistb:  # tlistb is usually like lost_tracks
        tid = t.track_id
        curr_tlbr = t.tlbr

        # Make sure there is no overlap
        do_skip = False
        for a_tlbr in tlista_tlbrs:
            area_overlap = find_area_overlap(a_tlbr, curr_tlbr)
            measured_overlap = area_overlap / find_area(curr_tlbr)

            if measured_overlap > 0.80:
                do_skip = True

        # Only add this to our joint list if we are not skipping
        if not exists.get(tid, 0) and not do_skip:
            exists[tid] = 1
            res.append(t)

    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
