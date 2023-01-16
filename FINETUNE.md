# Fine-tuning VideoMAE for Spatiotemporal Action Detection

The implementation supports **multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts folder](scripts/ava).

-  For example, to fine-tune VideoMAE ViT-B (**pre-trained & fine-tuned on k400**) on **AVA v2.2** with 32 GPUs (4 nodes x 8 GPUs), you can run

  ```bash
  # Set the path to save checkpoints and logs
  OUTPUT_DIR='YOUR_PATH/ava_videomae_vit_base_k400_pretrain+finetune'
  # path to pretrain model
  # Google Drive Link: 
  # https://drive.google.com/file/d/1MzwteHH-1yuMnFb8vRBQDvngV1Zl-d3z
  MODEL_PATH='YOUR_PATH_TO_PRETRAINED_MODEL/checkpoint.pth'

  # batch_size can be adjusted according to number of GPUs
  # this script is for 32 GPUs (4 nodes x 8 GPUs)
  OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12320 --nnodes=4 \
        --node_rank=$1 --master_addr=$2 run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 5e-4 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 30 \
        --data_set "ava" \
        --drop_path 0.2 \
        --val_freq 10 
  ```

  on the first node. On other nodes, run the same command with `--node_rank 1`, ..., `--node_rank 3` respectively.  `--master_addr` is set as the ip of the node 0.

  The results will be stored into `'YOUR_PATH/ava_videomae_vit_base_k400_pretrain+finetune/inference/result.log'`

  ```
  {'PascalBoxes_PerformanceByCategory/AP@0.5IOU/answer phone': 0.7145934868790218,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/bend/bow (at the waist)': 0.414221357927966,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/carry/hold (an object)': 0.5542964022180941,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/climb (e.g., a mountain)': 0.13882784458896855,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/close (e.g., a door, a box)': 0.19457154331541843,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/crouch/kneel': 0.2373069373690234,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/cut': 0.027696354157687696,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/dance': 0.5382088222382764,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/dress/put on clothing': 0.07400812186651312,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/drink': 0.2546625758263041,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/drive (e.g., a car, a truck)': 0.6385942316341753,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/eat': 0.2963977527682309,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/enter': 0.06496045047838869,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/fall down': 0.14899666849165968,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/fight/hit (a person)': 0.4689296719995923,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/get up': 0.3433543950782906,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/give/serve (an object) to (a person)': 0.12384097343247183,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/grab (a person)': 0.07055111509331946,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hand clap': 0.31080041475250836,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hand shake': 0.027005370122409115,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hand wave': 0.01865589206329232,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hit (an object)': 0.03373723374752993,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hug (a person)': 0.14587462158057204,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/jump/leap': 0.06419665393363826,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/kiss (a person)': 0.22093807137373472,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/lie/sleep': 0.45547708047674706,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/lift (a person)': 0.027453439884100535,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/lift/pick up': 0.0483429123577951,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/listen (e.g., to music)': 0.02160666510478387,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/listen to (a person)': 0.7128143796118426,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/martial art': 0.5159685575214354,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/open (e.g., a window, a car door)': 0.28566438094480723,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/play musical instrument': 0.38803198990043614,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/point to (an object)': 0.0014813193805880318,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/pull (an object)': 0.01668354978156843,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/push (an object)': 0.042630155731743835,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/push (another person)': 0.03221922403318834,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/put down': 0.02390227891014254,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/read': 0.293851638506127,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/ride (e.g., a bike, a car, a horse)': 0.4927320179043153,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/run/jog': 0.5582681877426154,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/sail boat': 0.21513809394285108,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/shoot': 0.0993969057243321,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/sing to (e.g., self, a person, a group)': 0.2017102619212805,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/sit': 0.8227546959555447,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/smoke': 0.10894415372399509,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/stand': 0.8525533362891095,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/swim': 0.4978887170831318,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/take (an object) from (a person)': 0.06026640716359491,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/take a photo': 0.004845860828248872,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/talk to (e.g., self, a person, a group)': 0.8219716274996891,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/text on/look at a cellphone': 0.012881372045325582,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/throw': 0.02376555272998438,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/touch (an object)': 0.35701735710818405,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/turn (e.g., a screwdriver)': 0.0053769276417869345,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/walk': 0.791870979746329,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (a person)': 0.7432475107259793,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (e.g., TV)': 0.17549066261544444,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/work on a computer': 0.0911826090742812,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/write': 0.09161035018829099,
  'PascalBoxes_Precision/mAP@0.5IOU': 0.2670044687122785}
  ```
### Note:

- Here total batch size = (`batch_size` per gpu) x `nodes` x (gpus per node).
- `lr` here is the base learning rate. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.
