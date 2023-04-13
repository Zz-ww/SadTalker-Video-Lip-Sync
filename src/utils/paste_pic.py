import cv2, os
import numpy as np
from tqdm import tqdm
import uuid
from src.inference_utils import Laplacian_Pyramid_Blending_with_mask


def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, restorer, enhancer, enhancer_region):
    video_stream_input = cv2.VideoCapture(pic_path)
    full_img_list = []
    while 1:
        input_reading, full_img = video_stream_input.read()
        if not input_reading:
            video_stream_input.release()
            break
        full_img_list.append(full_img)

    frame_h = full_img_list[0].shape[0]
    frame_w = full_img_list[0].shape[1]

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)

    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        clx, cly, crx, cry = crop_info[1]

    tmp_path = str(uuid.uuid4()) + '.mp4'
    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))

    for index, crop_frame in enumerate(tqdm(crop_frames, 'faceClone:')):
        p = cv2.resize(crop_frame.astype(np.uint8), (crx - clx, cry - cly))

        ff = full_img_list[index].copy()
        ff[cly:cry, clx:crx] = p
        if enhancer_region == 'none':
            pp = ff
        else:
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                ff, has_aligned=False, only_center_face=True, paste_back=True)
            if enhancer_region == 'lip':
                mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            else:
                mm = [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mouse_mask = np.zeros_like(restored_img)
            tmp_mask = enhancer.faceparser.process(restored_img[cly:cry, clx:crx], mm)[0]
            mouse_mask[cly:cry, clx:crx] = cv2.resize(tmp_mask, (crx - clx, cry - cly))[:, :, np.newaxis] / 255.

            height, width = ff.shape[:2]
            restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in
                                           (restored_img, ff, np.float32(mouse_mask))]
            img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
            pp = np.uint8(cv2.resize(np.clip(img, 0, 255), (width, height)))
            pp, orig_faces, enhanced_faces = enhancer.process(pp, full_img_list[index], bbox=[cly, cry, clx, crx],
                                                              face_enhance=False, possion_blending=True)
        out_tmp.write(pp)
    out_tmp.release()
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (tmp_path, new_audio_path, full_video_path)
    os.system(cmd)
    # os.remove(tmp_path)
    return tmp_path, new_audio_path
