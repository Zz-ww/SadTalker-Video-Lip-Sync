# SadTalker-Video-Lip-Sync


本项目基于SadTalkers实现视频唇形合成的Wav2lip。通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度。使用DAIN 插帧的DL算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然。

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
├──sync_show
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
                    --use_DAIN \ #(使用该功能会占用较大显存和消耗较多时间)
             		--time_step 0.5 #(插帧频率，默认0.5，即25fps—>50fps;0.25,即25fps—>100fps)
```



## 4.合成效果(Results)

```python
#合成效果展示在./sync_show目录下：
#original.mp4 原始视频
#sync_none.mp4 无任何增强的合成效果
#none_dain_50fps.mp4 只使用DAIN模型将25fps添帧到50fps
#lip_dain_50fps.mp4 对唇形区域进行增强使唇形更清晰+DAIN模型将25fps添帧到50fps
#face_dain_50fps.mp4 对全脸区域进行增强使唇形更清晰+DAIN模型将25fps添帧到50fps

#下面是不同方法的生成效果的视频
#our.mp4 本项目SadTalker-Video-Lip-Sync生成的视频
#sadtalker.mp4 sadtalker生成的full视频
#retalking.mp4 retalking生成的视频
#wav2lip.mp4 wav2lip生成的视频
```

https://user-images.githubusercontent.com/52994134/231769817-8196ef1b-c341-41fa-9b6b-63ad0daf14ce.mp4

视频拼接到一起导致帧数统一到25fps了，插帧效果看不出来区别，具体细节可以看./sync_show目录下的单个视频进行比较。

**本项目和sadtalker、retalking、wav2lip唇形合成的效果比较：**

|                           **our**                            |                        **sadtalker**                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <video  src="https://user-images.githubusercontent.com/52994134/233003969-91fa9e94-a958-4e2d-b958-902cc7711b8a.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003985-86d0f75c-d27f-4a52-ac31-2649ccd39616.mp4" type="video/mp4"> </video> |
|                        **retalking**                         |                         **wav2lip**                          |
| <video  src="https://user-images.githubusercontent.com/52994134/233003982-2fe1b33c-b455-4afc-ab50-f6b40070e2ca.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003990-2f8c4b84-dc74-4dc5-9dad-a8285e728ecb.mp4" type="video/mp4"> </video> |

readme中展示视频做了resize，原始视频可以看./sync_show目录下不同类别合成的视频进行比较。

## 5.预训练模型（Pretrained model）

预训练的模型如下所示：

```python
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├──auido2exp_00300-model.pth
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

百度网盘：https://pan.baidu.com/s/15-zjk64SGQnRT9qIduTe2A  提取码：klfv

谷歌网盘：https://drive.google.com/file/d/1lW4mf5YNtS4MAD7ZkAauDDWp2N3_Qzs7/view?usp=sharing

夸克网盘：https://pan.quark.cn/s/2a1042b1d046  提取码：zMBP

```python
#下载压缩包后解压到项目路径（谷歌网盘和夸克网盘下载的需要执行）
cd SadTalker-Video-Lip-Sync
tar -zxvf checkpoints.tar.gz
```

## 参考(Reference)

- SadTalker:https://github.com/Winfredy/SadTalker
-  VideoReTalking：https://github.com/vinthony/video-retalking
- DAIN :https://arxiv.org/abs/1904.00830
- PaddleGAN:https://github.com/PaddlePaddle/PaddleGAN
