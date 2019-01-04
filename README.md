<h1> Image captioning using Bottom-up, Top-down Attention</h1>

This is a PyTorch implementation of Image Captioning using Bottom-up, Top-down Attention as described <a href=http://www.panderson.me/up-down-attention>here</a>. Training and evaluation is done on the MSCOCO Image captioning challenge dataset. Bottom up features for MSCOCO dataset are extracted using Faster R-CNN object detection model trained on Visual Genome dataset. Pretrained bottom-up features are downloaded from <a href =https://github.com/peteanderson80/bottom-up-attention>here</a>.


<h3> Results obtained </h3> 

<table class="tg">
  <tr>
    <th>Model</th>
    <th>Epoch</th>
    <th>BLEU-4</th>
    <th>METEOR</th>
    <th>ROUGE-L</th>
    <th>CIDEr</th>
  </tr>
  <tr>
    <td><a target = "_blank" href="https://drive.google.com/open?id=19U83mLoMLnTOyKKkbA590WDqIo0srHIb">Bottom Up, Top Down Model with RELU gate</a></td>
    <td>27</td>
    <td>36.1</td>
    <td>27.2</td>
    <td>56.3</td>
    <td>112.4</td>
  </tr>
</table>

Results reported on Karpathy test split. Results almost similar to original paper. Pretrained model can be downloaded by clicking on the model link above.

<h3> Data preparation </h3>

Create a folder called 'data'

Create a folder called 'final_dataset'

Download the MSCOCO <a target = "_blank" href="http://images.cocodataset.org/zips/train2014.zip">Training</a> (13GB)  and <a href=http://images.cocodataset.org/zips/val2014.zip>Validation</a> (6GB)  images. 

Also download Andrej Karpathy's <a target = "_blank" href=http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip>training, validation, and test splits</a>. This zip file contains the captions.

Unzip all files and place the folders in 'data' folder.

<br>

Next, download the <a target = "_blank" href="https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip">bottom up image features</a>.

Unzip the folder and place unzipped folder in 'bottom-up_features' folder.  

<br>

Next type this command in a python 2 environment: 
```bash
python bottom-up_features/tsv.py
```

This command will create the following files - 
<ul>
<li>An HDF5 file containing the bottom up image features for train and val splits, 36 per image for each split, in an I, 36, 2048 tensor where I is the number of images in the split.</li>
<li>PKL files that contain training and validation image IDs mapping to index in HDF5 dataset created above.</li>
</ul>

Move these files to the folder 'final_dataset'.

<br>

Next, type this command: 
```bash
python create_input_files.py
```

This command will create the following files - 
<ul>
<li>A JSON file for each split containing the order in which to load the bottom up image features so that they are in lockstep with the captions loaded by the dataloader.</li>
<li>A JSON file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.</li>
<li>A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.</li>
<li>A JSON file which contains the word_map, the word-to-index dictionary.</li>
</ul>

<br>

Next, go to nlg_eval_master folder and type the following two commands:
```bash
pip install -e .
nlg-eval --setup
```
This will install all the files needed for evaluation.


<h3> Training </h3>

To train the bottom-up top down model from scratch, type:
```bash
python train.py
```

The dataset used for learning and evaluation is the MSCOCO Image captioning challenge dataset. It is split into training, validation and test sets using the popular Karpathy splits. This split contains 113,287 training images with five captions each, and 5K images respectively for validation and testing. Teacher forcing is used to aid convergence during training. Teacher forcing is a method of training sequence based tasks on recurrent neural networks by using the actual or expected output from the training dataset at the current time step y(t) as input in the next time step X(t+1), rather than the output generated by the network. Teacher forcing addresses slow convergence and instability when training recurrent networks that use model output from a prior time step as an input.

ReLU gate was used instead of TanH in the attention model.

Weight normalization was found to prevent the model from overfitting and is used liberally for all fully connected layers.

Gradients are clipped during training to prevent gradient explosion that is not uncommon with LSTMs. The attention dimensions, word embedding dimension and hidden dimensions of the LSTMs are set to 1024.

Dropout is set to 0.5. Batch size is set to 100. 36 pretrained bottom-up feature maps per image are used as input to the Top-down Attention model. The Adamax optimizer is used with a learning rate of 2e-3. Early stopping is employed if the BLEU-4 score of the validation set shows no improvement over 20 epochs.


<h3> Evaluation </h3>

To evaluate the model on the karpathy test split, edit the eval.py file to include the model checkpoint location and then type:
```bash
python eval.py
```

Beam search is used to generate captions during evaluation. Beam search iteratively considers the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keeps only the resulting best k of them. A beam search of five is used for inference.

The metrics reported are ones used most often in relation to image captioning and include BLEU-4, CIDEr, METEOR and ROUGE-L. Official MSCOCO evaluation scripts are used for measuring these scores.
  
<h3>References</h3>

Code adapted with thanks from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Evaluation code adapted from https://github.com/Maluuba/nlg-eval/tree/master/nlgeval

Tips for improving model performance and code for converting bottom-up features tsv file to hdf5 files sourced from https://github.com/hengyuan-hu/bottom-up-attention-vqa

https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/

