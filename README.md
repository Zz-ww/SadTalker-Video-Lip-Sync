# SadTalker-Video-Lip-Sync


本项目基于SadTalker做的视频唇形合成的Wav2lip。可以通过视频文件进行语音驱动唇形生成，修改了项目中的人脸增强的方式，添加了面部区域可配置的增强方式进行人脸唇形区域增强，加入DAIN模型对生成视频进行补帧，增强生成唇形的帧间流畅度，使驱动的唇形动作更为流畅和自然。

## 1.环境准备(Environment)

```python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install ffmpeg
pip install -r requirements.txt

#如需使用DAIN模型进行补帧需安装paddle
# CUDA 11.2
python -m pip install paddlepaddle-gpu==2.3.2.post112 \
-f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## 2.项目结构(Repository structure)

```
SadTalker-Video-Lip-Sync
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├── ...
├──dian_output
|   ├── ...
├──examples
|   ├── audio
|   ├── video
├──results
|   ├── ...
├──src
|   ├── ...
├──third_part
|   ├── ...
├──...
├──inference.py
├──README.md
```

## 3.模型推理(Inference)

```python
python inference.py --driven_audio <audio.wav> \
                    --source_video <video.mp4> \
                    --enhancer <none,lip,face> \  #(默认lip)
                    ----use_DAIN #(使用该功能会占用较大显存和消耗较多时间)
```



## 4.合成效果(Results)

```python
#合成效果展示在./sync_show目录下：
#original.mp4 原始视频
#sync_none.mp4 无任何增强的合成效果
#none_dain_50fps.mp4 只使用DAIN模型将25fps添帧到50fps
#lip_dain_50fps.mp4 对唇形区域进行增强使唇形更清晰+DAIN模型将25fps添帧到50fps
#face_dain_50fps.mp4 对全脸区域进行增强使唇形更清晰+DAIN模型将25fps添帧到50fps
```



## 5.预训练模型（Pretrained model）

预训练的模型如下所示：

```python
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├──auido2pose_00140-model.pth
|   ├──epoch_20.pth
|   ├──facevid2vid_00189-model.pth.tar
|   ├──GFPGANv1.3.pth
|   ├──GPEN-BFR-512.pth
|   ├──mapping_00109-model.pth.tar
|   ├──ParseNet-latest.pth
|   ├──RetinaFace-R50.pth
|   ├──shape_predictor_68_face_landmarks.dat
|   ├──wav2lip.pth
```

预训练的模型checkpoints下载路径:

夸克网盘：
链接：https://pan.quark.cn/s/2a1042b1d046
提取码：zMBP

百度网盘：（还在传速度太慢，后续更新）

谷歌网盘：（还在传速度太慢，后续更新）

```python
cd SadTalker-Video-Lip-Sync
tar -zxvf checkpoints.tar.gz
```



## 参考(Reference)

- SadTalker:https://github.com/Winfredy/SadTalker
-  VideoReTalking：https://github.com/vinthony/video-retalking
- DAIN :https://arxiv.org/abs/1904.00830
- PaddleGAN:https://github.com/PaddlePaddle/PaddleGAN
