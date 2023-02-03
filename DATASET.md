# Data Preparation

### Easy Version
We have successfully fine-tuned our VideoMAE on [AVA v2.2](https://research.google.com/ava/) with this codebase.

- The pre-processing of **AVA v2.2** can be summarized into 2 steps:

  1. Download the processed dataset from [Google Drive](https://drive.google.com/file/d/1lqDuz3zaP-wma3QbexDxtWW6stza5YFZ) or [Baidu NetDisk](https://pan.baidu.com/s/1UrflK4IgiVbVBOP5fDHdKA) (code `q5v5`).

  2. run following commands to unzip the file and create a symbolic link to the extracted files.
     ```
     tar zxvf AVA_compress.tar.gz -C /your/path/
     cd /path/to/VideoMAE-Action-Detection/
     ln -s /your/path/AVA data/AVA
     ```

### Step-by-step Version
Please follow the instructions in [AlphAction](https://github.com/MVIG-SJTU/AlphAction/blob/master/DATA.md#step-by-step-version) for step-by-step data preparation.
