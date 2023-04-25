import argparse
import h5py
import os
import sys
basepath = os.path.abspath(os.path.dirname(os.path.dirname('.')))
sys.path.append(basepath)
import numpy as np, os, scipy, scipy.spatial, matplotlib.pyplot as plt, IPython.display as ipd
# from numba import jit
import librosa
import libfmp.b, libfmp.c3, libfmp.c5
import pandas as pd, pickle, re
# from numba import jit
import torch.utils.data
import torch.nn as nn
import libdl.data_preprocessing
from libdl.data_loaders import dataset_context
from libdl.nn_models import basic_cnn_segm_sigmoid, deep_cnn_segm_sigmoid, simple_u_net_largekernels
from libdl.nn_models import simple_u_net_doubleselfattn, simple_u_net_doubleselfattn_twolayers
from libdl.nn_models import u_net_blstm_varlayers, simple_u_net_polyphony_classif_softmax
from libdl.data_preprocessing import compute_hopsize_cqt, compute_hcqt, compute_efficient_hcqt, compute_annotation_array_nooverlap
from torchinfo import summary


def load_model(model_name: str):
    """
    Initialize model weights and load parameters.
    """
    
    if model_name == 'unet':
        params_file = 'unet_params.json'

        fn_model = 'RETRAIN4_exp195f_musicnet_aligned_unet_extremelylarge_polyphony_softmax_rerun1.pt'

    else:
        params_file = 'cnn_params.json'
        
        if 'bigmix' in  model_name:
            fn_model = 'exp214c_bigmix_aligned_cnn_deepresnetwide.pt'
        else:
            fn_model = 'RETRAIN4_exp128c_musicnet_aligned_cnn_deepresnetwide_moresamples_rerun2.pt'
        
    with open(params_file, 'r') as f:
        import json
        
        mp = json.load(f)

    # init model Datastructure
    model = simple_u_net_polyphony_classif_softmax(n_chan_input=mp['n_chan_input'], n_chan_layers=mp['n_chan_layers'], \
        n_bins_in=mp['n_bins_in'], n_bins_out=mp['n_bins_out'], a_lrelu=mp['a_lrelu'], p_dropout=mp['p_dropout'], \
        scalefac=mp['scalefac'], num_polyphony_steps=mp['num_polyphony_steps'])

    # load weights of pretrained model
    dir_models = os.path.join(basepath, 'models_pretrained')
    path_trained_model = os.path.join(dir_models, fn_model)
    model.load_state_dict(torch.load(path_trained_model, map_location=torch.device('cpu')))
    model.eval()
    summary(model, input_size=(1, 6, 174, 216))
    
    return model


def gen_hcqts(inputdir: str):
    """
    Generator to extract HCQTs from MP3s in inputdir.
    """
    
    fs = 22050

    with open('hcqt_params.json', 'r') as f:
        import json
        
        hp = json.load(f)
    
    for root, dirs, files in os.walk(inputdir):
        for file in files:
            if file.endswith(".mp3"):
                
                print(file)
            
                fn_audio = os.path.join(root, file)

                path_audio = os.path.join(inputdir, fn_audio)
                f_audio, fs_load = librosa.load(path_audio, sr=fs)
            
                f_hcqt, fs_hcqt, hopsize_cqt = compute_efficient_hcqt(f_audio, fs=fs_load, fmin=librosa.note_to_hz('C1'), fs_hcqt_target=50, \
                                                                bins_per_octave=hp["bins_per_semitone"]*12, num_octaves=hp["num_octaves"], \
                                                                num_harmonics=hp["num_harmonics"], num_subharmonics=hp["num_subharmonics"], center_bins=hp["center_bins"])
                
                yield f_hcqt, fs_hcqt, hopsize_cqt, path_audio


def predict(f_hcqt, fs_hcqt, hopsize_cqt, model):
    """
    Extract multipitch features from the HCQT representation.
    """

    # Set test parameters
    test_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 1
                }
    device = 'cpu'

    test_dataset_params = {'context': 75,
                        'stride': 1,
                        'compression': 10
                        }
    
    half_context = test_dataset_params['context']//2

    inputs = np.transpose(f_hcqt, (2, 1, 0))
    targets = np.zeros(inputs.shape[1:]) # need dummy targets to use dataset object

    inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context+1), (0, 0))))
    targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context+1), (0, 0))))

    test_set = dataset_context(inputs_context, targets_context, test_dataset_params)
    test_generator = torch.utils.data.DataLoader(test_set, **test_params)

    pred_tot = np.zeros((0, model.n_bins_out))

    max_frames = 160
    k=0
    for test_batch, test_labels in test_generator:
        k+=1
        if k>max_frames:
            break
        # Model computations
        y_pred, n_pred = model(test_batch)
        pred_log = torch.squeeze(torch.squeeze(y_pred.to('cpu'),2),1).detach().numpy()
        # pred_log = torch.squeeze(y_pred.to('cpu')).detach().numpy()
        pred_tot = np.append(pred_tot, pred_log, axis=0)
        
    return pred_tot  
    
    
    
    

def main(inputdir: str, model_name: str):
    
    model = load_model(model_name=model_name)
    
    outdir = inputdir.replace("audiodata", f"multipitch/{model}/")
    os.makedirs(outdir, exist_ok=False)
    
    for f_hcqt, fs_hcqt, hopsize_cqt, path_audio in gen_hcqts():
        
        preds = predict(f_hcqt, fs_hcqt, hopsize_cqt, model)
        
        with h5py.File(path_audio.replace("audiodata", f"multipitch/{model_name}/"), "w") as f:
            
            f.create_dataset(name='multipitch', data=preds)
            
            
            
            
            
        
        
        
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract multipitch features from MP3 file directory.")
    parser.add_argument('-i', '--inputdir', type=str, help='Input directory which is recursively searched containing the MP3 files.')
    parser.add_argument('-m', '--model', type=str, choices=['unet', 'cnn', 'cnn_bigmix'], default='unet', help='The model to use for extraction.')
    args = parser.parse_args()
    main(args.inputdir, args.model)