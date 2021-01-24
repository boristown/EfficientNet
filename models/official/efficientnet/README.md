# EfficientNets

[1] Mingxing Tan and Quoc V. Le.  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.
   Arxiv link: https://arxiv.org/abs/1905.11946.

Updates

  - **[Mar 2020] Released mobile/IoT device friendly EfficientNet-lite models: [README](lite/README.md).**

  - [Feb 2020] Released EfficientNet checkpoints trained with NoisyStudent: [paper](https://arxiv.org/abs/1911.04252).

  - [Nov 2019] Released EfficientNet checkpoints trained with AdvProp: [paper](https://arxiv.org/abs/1911.09665).

  - [Oct 2019] Released EfficientNet-CondConv models with conditionally parameterized convolutions: [README](condconv/README.md), [paper](https://arxiv.org/abs/1904.04971).

  - [Oct 2019] Released EfficientNet models trained with RandAugment: [paper](https://arxiv.org/abs/1909.13719).

  - [Aug 2019] Released EfficientNet-EdgeTPU models: [README](edgetpu/README.md) and [blog post](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html).

  - [Jul 2019] Released EfficientNet checkpoints trained with AutoAugment: [paper](https://arxiv.org/abs/1805.09501), [blog post](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html)

  - [May 2019] Released EfficientNets code and weights: [blog post](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)

## 1. About EfficientNet Models

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use [AutoML MNAS Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="./g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="./g3doc/flops.png", width="100%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, our EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.

## 2. Using Pretrained EfficientNet Checkpoints

To train EfficientNet on ImageNet, we hold out 25,022 randomly picked images ([image filenames](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/val_split20.txt), or 20 out of 1024 total shards) as a 'minival' split, and conduct early stopping based on this 'minival' split. The final accuracy is reported on the original ImageNet validation set.

We have provided a list of EfficientNet checkpoints:.

  * With baseline ResNet preprocessing, we achieve similar results to the original ICML paper.
  * With [AutoAugment](https://arxiv.org/abs/1805.09501) preprocessing, we achieve higher accuracy than the original ICML paper.
  * With [RandAugment](https://arxiv.org/abs/1909.13719) preprocessing, accuracy is further improved.
  * With [AdvProp](https://arxiv.org/abs/1911.09665), state-of-the-art results (w/o extra data) are achieved.
  * With [NoisyStudent](https://arxiv.org/abs/1911.04252), state-of-the-art results (w/ extra JFT-300M unlabeled data) are achieved.

|               |   B0    |  B1   |  B2    |  B3   |  B4   |  B5    | B6 | B7 | B8 | L2-475 | L2 |
|----------     |--------  | ------| ------|------ |------ |------ | --- | --- | --- | --- |--- |
| Baseline preprocessing |  76.7% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b0.tar.gz))   | 78.7% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b1.tar.gz))  | 79.8% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b2.tar.gz)) | 81.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b3.tar.gz)) | 82.5% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b4.tar.gz)) | 83.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b5.tar.gz)) | | || | | |
| AutoAugment (AA) |  77.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz))   | 79.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b1.tar.gz))  | 80.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b2.tar.gz)) | 81.6% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b3.tar.gz)) | 82.9% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b4.tar.gz)) | 83.6% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b5.tar.gz)) |  84.0% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b6.tar.gz)) | 84.3% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b7.tar.gz))  || | |
| RandAugment (RA) |  |  |  |  |  | 83.7%  ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/randaug/efficientnet-b5-randaug.tar.gz)) |  | 84.7%  ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/randaug/efficientnet-b7-randaug.tar.gz)) |  | | |
| AdvProp + AA | 77.6% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b0.tar.gz)) | 79.6% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b1.tar.gz))  | 80.5% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b2.tar.gz)) | 81.9% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b3.tar.gz)) | 83.3% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b4.tar.gz)) | 84.3% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b5.tar.gz)) | 84.8% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b6.tar.gz)) | 85.2% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b7.tar.gz)) | 85.5% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b8.tar.gz))|| | |
| NoisyStudent + RA | 78.8% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b0.tar.gz)) | 81.5% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b1.tar.gz)) | 82.4% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b2.tar.gz)) | 84.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b3.tar.gz)) | 85.3% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b4.tar.gz)) | 86.1% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b5.tar.gz)) | 86.4% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b6.tar.gz)) | 86.9% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b7.tar.gz)) | - |88.2%([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-l2_475.tar.gz))|88.4% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-l2.tar.gz)) | 

<!--
| Acc. from paper        |  76.3%   | 78.8% | 79.8% | 81.1% | 82.6% | 83.3% |
-->

<sup>*To train EfficientNets with AutoAugment ([code](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)), simply add option "--augment_name=autoaugment". If you use these checkpoints, you can cite this [paper](https://arxiv.org/abs/1805.09501).</sup>

<sup>**To train EfficientNets with RandAugment ([code](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)), simply add option "--augment_name=randaugment". For EfficientNet-B5 also add "--randaug_num_layers=2 --randaug_magnitude=17". For EfficientNet-B7 or EfficientNet-B8 also add "--randaug_num_layers=2 --randaug_magnitude=28". If you use these checkpoints, you can cite this [paper](https://arxiv.org/abs/1909.13719).</sup>

<sup>* AdvProp training code coming soon. Please set "--advprop_preprocessing=True" for using AdvProp checkpoints.  If you use AdvProp checkpoints, you can cite this [paper](https://arxiv.org/abs/1911.09665).</sup>

<sup>* NoisyStudent training code coming soon. L2-475 means the same L2 architecture with input image size 475 (Please set "--input_image_size=475" for using this checkpoint). If you use NoisyStudent checkpoints, you can cite this [paper](https://arxiv.org/abs/1911.04252).</sup>

<sup>*Note that AdvProp and NoisyStudent performance is derived from baselines that don't use holdout eval set. They will be updated in future."</sup>

A quick way to use these checkpoints is to run:

    $ export MODEL=efficientnet-b0
    $ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/${MODEL}.tar.gz
    $ tar xf ${MODEL}.tar.gz
    $ wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
    $ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.json
    $ python eval_ckpt_main.py --model_name=$MODEL --ckpt_dir=$MODEL --example_img=panda.jpg --labels_map_file=labels_map.json

Please refer to the following colab for more instructions on how to obtain and use those checkpoints.

  * [`eval_ckpt_example.ipynb`](eval_ckpt_example.ipynb): A colab example to load
 EfficientNet pretrained checkpoints files and use the restored model to classify images.


## 3. Using EfficientNet as Feature Extractor

```
    import efficientnet_builder
    features, endpoints = efficientnet_builder.build_model_base(images, 'efficientnet-b0')
```

  * Use `features` for classification finetuning.
  * Use `endpoints['reduction_i']` for detection/segmentation, as the last intermediate feature with reduction level `i`. For example, if input image has resolution 224x224, then:
    * `endpoints['reduction_1']` has resolution 112x112
    * `endpoints['reduction_2']` has resolution 56x56
    * `endpoints['reduction_3']` has resolution 28x28
    * `endpoints['reduction_4']` has resolution 14x14
    * `endpoints['reduction_5']` has resolution 7x7

## 4. Training EfficientNets on TPUs.


To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource
   * A GCS bucket to store your training checkpoints (the "model directory")
   * Install TensorFlow version >= 1.13 for both GCE VM and Cloud.

Then train the model:

    $ export PYTHONPATH="$PYTHONPATH:/path/to/models"
    $ python main.py --tpu=TPU_NAME --data_dir=DATA_DIR --model_dir=MODEL_DIR

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions, please refer to our tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet

## 5. Guides


export PROJECT_NAME=hellotpuresnet50
gcloud config set project $PROJECT_NAME

ctpu up -name=zeroaitpu -tpu-size=v3-8
ctpu up -name=zeroaitpu -tpu-size=v2-8
ctpu up -preemptible -name=zeroaitpu -tpu-size=v3-8
ctpu up -preemptible -name=zeroaitpu -tpu-size=v2-8

export PROJECT_NAME=hellotpuresnet50
gcloud config set project $PROJECT_NAME
ctpu up -name=zeroaitpu -tpu-size=v3-8 -zone=us-central1-a --project $PROJECT_NAME
ctpu up -preemptible -name=zeroaitpu -tpu-size=v3-8 -zone=us-central1-a --project $PROJECT_NAME

ctpu up -preemptible -name=zeroaitpu -tpu-size=v2-8

ctpu up -preemptible -name=zeroaitpu -tpu-size=v3-8

pip install tensorflow
export STORAGE_BUCKET=gs://zeroaistorage 
export PYTHONPATH="$PYTHONPATH:/boristown/models" 
cd ~/ 
rm -rf boristown
git clone https://github.com/boristown/tpu.git boristown 
cd boristown/models/official/resnet/
python resnet_main.py --train_steps=50400 --train_batch_size=40000 --eval_batch_size=40000 --num_train_images=125062534 --num_eval_images=2480700 --steps_per_eval=500 --iterations_per_loop=500 --dropblock_groups="" --dropblock_keep_prob="1" --dropblock_size="1" --resnet_depth=201 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=20400 --train_batch_size=40000 --eval_batch_size=40000 --num_train_images=154034586 --num_eval_images=3480050 --steps_per_eval=1000 --iterations_per_loop=1000 --dropblock_groups="" --dropblock_keep_prob="1" --dropblock_size="1" --resnet_depth=201 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=10200 --train_batch_size=50000 --eval_batch_size=50000 --num_train_images=154034586 --num_eval_images=3480050 --steps_per_eval=300 --iterations_per_loop=300 --dropblock_groups="" --dropblock_keep_prob="1" --dropblock_size="1" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=4000 --train_batch_size=40000 --eval_batch_size=40000 --num_train_images=158399462 --num_eval_images=2833734 --steps_per_eval=50 --iterations_per_loop=200 --dropblock_groups="" --dropblock_keep_prob="1" --dropblock_size="1" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=6100 --train_batch_size=20000 --eval_batch_size=20000 --num_train_images=91552720 --num_eval_images=1881360 --steps_per_eval=300 --iterations_per_loop=300 --dropblock_groups="" --dropblock_keep_prob="0.9" --dropblock_size="1" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=72261 --train_batch_size=8 --eval_batch_size=8 --num_train_images=578094 --num_eval_images=2432 --steps_per_eval=100 --iterations_per_loop=100 --dropblock_groups="" --dropblock_keep_prob="0.5" --dropblock_size="3" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="float32" --data_format="channels_last" 

python resnet_main.py --train_steps=9000 --train_batch_size=102712 --eval_batch_size=101416 --num_train_images=924432509 --num_eval_images=4346738 --steps_per_eval=450 --iterations_per_loop=450 --dropblock_groups="" --dropblock_keep_prob="0.5" --dropblock_size="3" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=29031744 --train_batch_size=103680 --eval_batch_size=101832 --num_train_images=29031744 --num_eval_images=678882 --steps_per_eval=420 --iterations_per_loop=420 --dropblock_groups="" --dropblock_keep_prob="0.5" --dropblock_size="3" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 

python resnet_main.py --train_steps=46048794 --train_batch_size=109632 --eval_batch_size=109480 --num_train_images=46048794 --num_eval_images=919672 --steps_per_eval=420 --iterations_per_loop=420 --dropblock_groups="" --dropblock_keep_prob="0.5" --dropblock_size="3" --resnet_depth=169 --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last" 


export PROJECT_NAME=hellotpuresnet50
gcloud config set project $PROJECT_NAME
ctpu up -preemptible -name=zeroaitpu -tpu-size=v3-8
export STORAGE_BUCKET=gs://zeroaistorage 
capture_tpu_profile --tpu=${TPU_NAME} --logdir=${STORAGE_BUCKET}/resnet
tensorboard --logdir=${STORAGE_BUCKET}/resnet

ctpu delete -name=zeroaitpu

sudo apt install unzip
export STORAGE_BUCKET=gs://zeroaistorage
unzip -n ${STORAGE_BUCKET}/6028.zip -d ${STORAGE_BUCKET}/data


=============================================================
rm -rf serving
git clone https://github.com/boristown/serving.git serving 

# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# Start TensorFlow Serving container and open the REST API port
sudo docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_turtle5:/models/turtle5" \
    -e MODEL_NAME=turtle5 \
    tensorflow/serving &

TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

sudo docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_turtle5:/models/turtle5" \
    -e MODEL_NAME=turtle5 \
    tensorflow/serving &
	
sudo systemctl restart apiturtle
sudo systemctl enable apiturtle
sudo systemctl status apiturtle

sudo systemctl restart nginx

sudo service apiturtle restart
sudo service nginx restart

sudo docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/models/:/models/" tensorflow/serving \
    --model_config_file=/models/models.config \
    --model_config_file_poll_wait_seconds=60
	
sudo docker run \
-p 8501:8501 \
--mount type=bind,source=/home/boristown/serving/tensorflow_serving/servables/tensorflow/testdata/,target=/models/ \
-t tensorflow/serving \
--model_config_file=/models/models.config \
--model_config_file_poll_wait_seconds=600

sudo nano /models/models.config
sudo nano serving/tensorflow_serving/servables/tensorflow/testdata/models.config

model_config_list {
  config {
    name: 'turtle3'
    base_path: '/home/boristown/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_turtle3/'
    model_platform: 'tensorflow'
  }
  config {
    name: 'turtle5'
    base_path: '/home/boristown/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_turtle5/'
    model_platform: 'tensorflow'
  }
}

model_config_list {
  config {
    name: 'turtle7'
    base_path: '/models/saved_model_turtle7/'
    model_platform: 'tensorflow'
  }
  config {
    name: 'turtle8'
    base_path: '/models/saved_model_turtle8/'
    model_platform: 'tensorflow'
  }
}

sudo ln -s /serving /models

sudo certbot certonly --nginx
==========================================================V2-512

ctpu up -preemptible -name=zeroaitpu -tpu-size=v2-512 -zone=us-central1-a

export STORAGE_BUCKET=gs://zeroaistorage 
export PYTHONPATH="$PYTHONPATH:/boristown/models" 
cd ~/ 
rm -rf boristown  
git clone https://github.com/boristown/tpu.git boristown 
cd boristown/models/official/resnet/
python resnet_main.py --num_cores=512 --train_steps=4417748 --train_batch_size=4000000 --eval_batch_size=4000000 --num_train_images=4417748 --num_eval_images=4417748 --steps_per_eval=200 --iterations_per_loop=200 --resnet_depth=50 --data_dir=${STORAGE_BUCKET}/data44216 --model_dir=${STORAGE_BUCKET}/resnet --tpu=${TPU_NAME} --precision="bfloat16" --data_format="channels_last"

ctpu up -preemptible -name=zeroaitpu -tpu-size=v2-512 -zone=us-central1-a
export STORAGE_BUCKET=gs://zeroaistorage 
capture_tpu_profile --tpu=${TPU_NAME} --logdir=${model_dir}/resnet 

tensorboard --logdir=${STORAGE_BUCKET}/resnet 

ctpu delete -name=zeroaitpu
