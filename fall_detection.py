import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FallDetecion():
    def __init__(self, threshold_fallscore = 40):
        self.threshold_fallscore = threshold_fallscore

    def __call__(self, skeleton_cache):
        
        last_angles = self.get_angles(skeleton_cache[-1])
        first_angles = self.get_angles(skeleton_cache[0])
        
        fallScore = 0
        j = 0
        
        for i in range(len(first_angles)):
            if first_angles[i] is None or last_angles[i] is None:
                continue
                
            j += 1
            fallScore += abs(last_angles[i] - first_angles[i])
        
        if j:
            fallScore /= j
       
        isFall = fallScore > self.threshold_fallscore

        print("Is Fall:", isFall)
        print("Fall Score:", fallScore,"\n")
        
        return isFall, fallScore 
    
    
    def get_angles(self, keypoints):
        return (
            self.compute_angle(
                keypoints[7],
                keypoints[5],
                keypoints[11]
            ),
            self.compute_angle(
                keypoints[8],
                keypoints[6],
                keypoints[12]
            ),
            self.compute_angle(
                keypoints[11],
                keypoints[13],
                keypoints[15]
            ),
            self.compute_angle(
                keypoints[12],
                keypoints[14],
                keypoints[16]
            ),
            self.compute_angle(
                keypoints[13],
                keypoints[11],
                keypoints[12]
            ),
            self.compute_angle(
                keypoints[14],
                keypoints[12],
                keypoints[11]
            ),
            self.compute_angle(
                keypoints[0],
                (keypoints[1][0]/2+keypoints[2][0]/2, keypoints[1][1]/2+keypoints[2][1]/2),
                (keypoints[0][0], keypoints[0][1]+100)
            ),
            self.compute_angle(
                keypoints[5],
                keypoints[11],
                (keypoints[5][0], keypoints[11][1])
            ),
            self.compute_angle(
                keypoints[6],
                keypoints[12],
                (keypoints[6][0], keypoints[12][1])
            ),
        )
    
    def compute_angle(self, point1, point2, point3):

        if np.all(point1 == 0) or np.all(point2 == 0) or np.all(point3 == 0):
            return None  

        point1 = point1[:2]
        point2 = point2[:2]
        point3 = point3[:2]
        
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        cosine_angle = min(1.0, max(-1.0, cosine_angle))

        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    