from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred * idx_tensor, 1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
                           torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                           torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
                         torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                         -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
                          torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                          torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat


def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']  # (bs, k, 3)
    yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp = exp * 0

        # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0] * 0
    t[:, 2] = t[:, 2] * 0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}


def make_animation(source_image, source_semantics, target_semantics, generator, kp_detector, mapping, frame_num):
    with torch.no_grad():
        predictions = []
        kp_canonical = []
        kp_source = []
        for single_image in source_image:
            kp_single = kp_detector(single_image)
            kp_canonical.append(kp_single)
            he_source_single = mapping(source_semantics)
            kp_single = keypoint_transformation(kp_single, he_source_single)
            kp_source.append(kp_single)

        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            if frame_idx >= frame_num:
                break
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            kp_driving = keypoint_transformation(kp_canonical[frame_idx], he_driving)
            kp_norm = kp_driving
            out = generator(source_image[frame_idx], kp_source=kp_source[frame_idx], kp_driving=kp_norm)
            predictions.append(out['prediction'])
        predictions_ts = torch.stack(predictions, dim=1)
    return predictions_ts
