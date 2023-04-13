import numpy as np
import cv2


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [gpA[num_levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        gm = gm[:, :, np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_
