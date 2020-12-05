<h1> Image captioning using Bottom-up, Top-down Attention</h1>

This is the PyTorch implementation of <a href=https://arxiv.org/abs/2009.12313> Are scene graphs good enough to improve Image Captioning?</a>. Training and evaluation is done on the MSCOCO Image captioning challenge dataset. Bottom up features for MSCOCO dataset are extracted using Faster R-CNN object detection model trained on Visual Genome dataset. Pretrained bottom-up features are downloaded from <a href =https://github.com/peteanderson80/bottom-up-attention>here</a>. 

This Repository is designed with every different model design in a different branch. The name of the branch indicates what the model design is. Iti s best to avoid the Main branch currently, since this is outdated. 

<h4> TODO: Clean Up the codebase to be contained in a single main branch. Planned to work on this, during the summer holidays </h4>

<!-- <h2> Results obtained </h2> 

<table class="tg">
  <tr>
    <th>Model</th>
    <th>BLEU-4</th>
    <th>METEOR</th>
    <th>ROUGE-L</th>
    <th>CIDEr</th>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/10atC8rY7PdhnKW08INO33mEXYUyQ6G0N/view?usp=sharing">This implementation</a></td>
    <td>35.9</td>
    <td>26.9</td>
    <td>56.2</td>
    <td>111.5</td>
  </tr>
  <tr>
    <td>Original paper implementation</td>
    <td>36.2</td>
    <td>27.0</td>
    <td>56.4</td>
    <td>113.5</td>
    </tr>
</table>

Results reported on Karpathy test split. Pretrained model can be downloaded by clicking on the link above.

<h2> Requirements </h2>

python 3.6<br>
torch 0.4.1<br>
h5py 2.8<br>
tqdm 4.26<br>
nltk 3.3<br> -->

<h2> Data preparation </h2>

Create a folder called 'data'

Create a folder called 'final_dataset'

Download the MSCOCO <a target = "_blank" href="http://images.cocodataset.org/zips/train2014.zip">Training</a> (13GB)  and <a href=http://images.cocodataset.org/zips/val2014.zip>Validation</a> (6GB)  images. 

Also download Andrej Karpathy's <a target = "_blank" href=http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip>training, validation, and test splits</a>. This zip file contains the captions.

Unzip all files and place the folders in 'data' folder.

<br>

Next, download the <a target = "_blank" href="https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip">bottom up image features</a>. We used the fixed 36 regions version.

Unzip the folder and place unzipped folder in 'bottom-up_features' folder.  

<br>

Next type this command in a python environment: 
```bash
python bottom-up_features/tsv.py
```

This command will create the following files - 
<ul>
<li>An HDF5 file containing the bottom up image features for train and val splits, 36 per image for each split, in an I, 36, 2048 tensor where I is the number of images in the split.</li>
<li>PKL files that contain training and validation image IDs mapping to index in HDF5 dataset created above.</li>
</ul>

<br>

optionally for the scene graphs, also run the following: 
```bash
python create_input_files.py
```

this will create the following similar HDF5 and PKL files. 

Move these files to the folder 'final_dataset'.

<br>

Next, type this command. If you dont want to prepare the scene-graph features, remove the -s flag: 
```bash
python create_input_files.py -s
```

This command will create the following files - 
<ul>
<li>A JSON file for each split containing the order in which to load the bottom up image features so that they are in lockstep with the captions loaded by the dataloader.</li>
<li>A JSON file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.</li>
<li>A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.</li>
<li>A JSON file which contains the word_map, the word-to-index dictionary.</li>
</ul>

<br>

Although we make use of the official COCO captioning evaluation scripts, for legacy kept the nl_eval_master folder. 

Next, go to nlg_eval_master folder and type the following two commands:
```bash
pip install -e .
nlg-eval --setup
```
This will install all the files needed for evaluation. 


<h2> Training </h2>

To train the bottom-up top down model, type:
```bash
python train.py
```

<h2> Evaluation </h2>

To evaluate the model on the karpathy test split, edit the eval.py file to include the model checkpoint location and then type:
```bash
python eval.py
```

Beam search is used to generate captions during evaluation. Beam search iteratively considers the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keeps only the resulting best k of them. A beam search of five is used for inference.

The metrics reported are ones used most often in relation to image captioning and include BLEU-4, CIDEr, METEOR and ROUGE-L. Official MSCOCO evaluation scripts are used for measuring these scores.
  
<h2>References</h2>

Code adapted with thanks from https://github.com/poojahira/image-captioning-bottom-up-top-down
