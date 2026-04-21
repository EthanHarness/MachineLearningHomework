from typing import List

import numpy as np
import numpy.typing as npt

class KMeansPlusPlus:
    def __init__(self, scaledInput: npt.NDArray[np.float64], iterations: int, kMeanCount: int) -> None:
        self.trData: npt.NDArray[np.float64] = scaledInput
        self.iterations: int = iterations
        self.kMeanCount: int = kMeanCount
        self.rng = np.random.default_rng()
        self.clusterCenters: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

        self.results: npt.NDArray[np.float64] = self.train()

    def getNextCenter(self, centers: List[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        distances: npt.NDArray[np.float64] = np.min(np.linalg.norm(self.trData[:, np.newaxis] - centers, axis=2), axis=1)
        distribution: npt.NDArray[np.float64] = distances**2
        distribution /= distribution.sum()
        index: int = self.rng.choice(len(self.trData), p=distribution)
        return self.trData[index]
        

    def train(self) -> npt.NDArray[np.float64]:
        results: npt.NDArray[np.float64] = np.array([0.0]*self.iterations)

        for x in range(self.iterations): 
            #Initialize centers
            centers: List[npt.NDArray[np.float64]] = [self.rng.choice(self.trData, size=1, replace=True, axis=0)[0]]
            for _ in range(self.kMeanCount - 1):
                centers.append(self.getNextCenter(centers))
            self.clusterCenters = np.array(centers)

            while(True):
                oldCenters: npt.NDArray[np.float64] = self.clusterCenters.copy()

                #Find closest center
                distances = np.linalg.norm(self.trData[:, np.newaxis] - self.clusterCenters, axis=2)
                labels = np.argmin(distances, axis=1)
                
                #Adjust center
                for i in range(self.kMeanCount):
                    points_in_cluster = self.trData[labels == i]
                    if len(points_in_cluster) > 0:
                        self.clusterCenters[i] = np.mean(points_in_cluster, axis=0)

                if np.allclose(oldCenters, self.clusterCenters, atol=1e-6, rtol=1e-2):
                    break

            assigned_centers = self.clusterCenters[labels]
            inertia = np.sum((self.trData - assigned_centers)**2)
            results[x] = inertia

        return results
    
    def getTrainingResults(self) -> npt.NDArray[np.float64]:
        return self.results

            


