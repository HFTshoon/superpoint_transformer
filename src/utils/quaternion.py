# change quaternion to rotation matrix

import torch
import numpy as np

def qvec2rotmat(qvec):
	rotmat = np.array([
		[1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
		2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
		2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
		[2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
		1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
		2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
		[2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
		2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
		1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
	return torch.tensor(rotmat, dtype=torch.float32)