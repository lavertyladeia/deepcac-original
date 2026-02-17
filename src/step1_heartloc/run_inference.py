"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 2.7.17
  ----------------------------------------
  
"""

import os

# Configure cuDNN environment variables before importing TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tables
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc  # For garbage collection to free GPU memory

from scipy.ndimage import measurements

import heartloc_model

def save_png(patientID, output_dir_png, img, msk, pred):
  maskIndicesMsk = np.where(msk != 0)
  if len(maskIndicesMsk) == 0:
    trueBB = [np.min(maskIndicesMsk[0]), np.max(maskIndicesMsk[0]),
              np.min(maskIndicesMsk[1]), np.max(maskIndicesMsk[1]),
              np.min(maskIndicesMsk[2]), np.max(maskIndicesMsk[2])]
    cen = [trueBB[0] + (trueBB[1] - trueBB[0]) / 2,
           trueBB[2] + (trueBB[3] - trueBB[2]) / 2,
           trueBB[4] + (trueBB[5] - trueBB[4]) / 2]
  else:
    cen = [int(len(img) / 2), int(len(img) / 2), int(len(img) / 2)]

  pred[pred > 0.5] = 1
  pred[pred < 1] = 0

  fig, ax = plt.subplots(2, 3, figsize=(32, 16))
  ax[0, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[0, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[0, 2].imshow(img[:, :, cen[2]], cmap='gray')

  ax[0, 0].imshow(msk[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[0, 1].imshow(msk[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[0, 2].imshow(msk[:, :, cen[2]], cmap='jet', alpha=0.4)

  ax[1, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[1, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[1, 2].imshow(img[:, :, cen[2]], cmap='gray')

  ax[1, 0].imshow(pred[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[1, 1].imshow(pred[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[1, 2].imshow(pred[:, :, cen[2]], cmap='jet', alpha=0.4)

  fileName = os.path.join(output_dir_png, patientID + '_' + ".png")
  plt.savefig(fileName)
  plt.close(fig)


def test(model, dataDir, output_dir_npy, output_dir_png, pkl_file,
         test_file, weights_file, mgpu, has_manual_seg, png):
    
  # Weights are already loaded in run_inference, so we don't need to load them again here

  testFileHdf5 = tables.open_file(os.path.join(dataDir, test_file), "r")
  pklData = pickle.load(open(os.path.join(dataDir, pkl_file), 'rb'))

  # Get data in one list for further processing
  testDataRaw = []
  num_test_imgs = len(testFileHdf5.root.ID)
  for i in range(num_test_imgs):
    patientID = testFileHdf5.root.ID[i]
    img = testFileHdf5.root.img[i]
    if has_manual_seg:
      msk = testFileHdf5.root.msk[i]
    else:  # Create empty dummy has_manual_seg with same size as the image
      sizeImg = len(img)
      msk = np.zeros((sizeImg, sizeImg, sizeImg), dtype=np.float64)
    if not patientID in pklData.keys():
      print('Patient not found in pkl data', patientID)
      continue
    zDif = pklData[patientID][6][2]
    testDataRaw.append([patientID, img, msk, zDif])

  numData = len(testDataRaw)
  size = len(testDataRaw[0][1])
  imgsTrue = np.zeros((numData, size, size, size), dtype=np.float64)
  msksTrue = np.zeros((numData, size, size, size), dtype=np.float64)

  # Process one image at a time to avoid GPU memory issues
  # This is safer for large volumes (112x112x112) and limited GPU memory
  batch_size = 1  # Process one patient at a time to avoid memory overflow
  
  for i in xrange(0, len(testDataRaw), batch_size):
    # Process in smaller batches to avoid memory issues
    current_batch_size = min(batch_size, len(testDataRaw) - i)
    imgTest = np.zeros((current_batch_size, size, size, size), dtype=np.float64)

    for j in range(current_batch_size):
      patientIndex = i + j
      if patientIndex >= len(testDataRaw):
        break
      patientID = testDataRaw[patientIndex][0]
      print('Processing patient', patientID)
      # Store data for score calculation
      imgsTrue[patientIndex, :, :, :] = testDataRaw[patientIndex][1]
      msksTrue[patientIndex, :, :, :] = testDataRaw[patientIndex][2]
      imgTest[j, :, :, :] = testDataRaw[patientIndex][1]

    # Predict with smaller batch to avoid GPU memory overflow
    try:
      msksPred = model.predict(imgTest[:, :, :, :, np.newaxis], batch_size=current_batch_size, verbose=0)
    except Exception as e:
      print('Error during prediction, trying to clear GPU memory and retry...')
      # Try to clear GPU memory
      import tensorflow.keras.backend as K  # type: ignore
      K.clear_session()
      # Retry with single image
      if current_batch_size > 1:
        print('Retrying with batch size 1...')
        imgTest_single = imgTest[0:1, :, :, :, np.newaxis]
        msksPred = model.predict(imgTest_single, batch_size=1, verbose=0)
        msksPred = np.expand_dims(msksPred, axis=0)  # Add batch dimension back
      else:
        raise e

    for j in range(current_batch_size):
      patientIndex = i + j
      if patientIndex >= len(testDataRaw):
        break
      patientID = testDataRaw[patientIndex][0]
      np.save(os.path.join(output_dir_npy, patientID + '_pred'),
              [[patientID], imgsTrue[patientIndex], msksTrue[patientIndex], msksPred[j, :, :, :, 0]])

    if png:
      for j in range(current_batch_size):
        patientIndex = i + j
        if patientIndex >= len(testDataRaw):
          break
        patientID = testDataRaw[patientIndex][0]
        save_png(patientID, output_dir_png, imgsTrue[patientIndex], msksTrue[patientIndex], msksPred[j, :, :, :, 0])
    
    # Clear memory after each batch to prevent accumulation
    del imgTest, msksPred
    gc.collect()


def run_inference(model_output_dir_path, model_input_dir_path, model_weights_dir_path,
                  crop_size, export_png, model_down_steps, extended, has_manual_seg, weights_file_name, num_gpus=1):

  if num_gpus == 1:
    print("\nDeep Learning model inference using 1 GPU:")
  else:
    print("\nDeep Learning model inference using {} GPUs:".format(num_gpus))
  
  mgpu = num_gpus

  output_dir_npy = os.path.join(model_output_dir_path, 'npy')
  output_dir_png = os.path.join(model_output_dir_path, 'png')
  if not os.path.exists(output_dir_npy):
    os.mkdir(output_dir_npy)
  if export_png and not os.path.exists(output_dir_png):
    os.mkdir(output_dir_png)

  test_file = "step1_test_data.h5"
  pkl_file = "step1_downsample_results.pkl"

  weights_file = os.path.join(model_weights_dir_path, weights_file_name)

  print('Loading saved model from "%s"'%(weights_file,))
  
  input_shape = (crop_size, crop_size, crop_size, 1)
  model = heartloc_model.get_unet_3d(down_steps = model_down_steps,
                                     input_shape = input_shape,
                                     mgpu = mgpu,
                                     ext = extended)

  # Load weights - handle both single GPU and multi-GPU saved weights
  # If weights were saved from a multi-GPU model, we need to load by name
  try:
    model.load_weights(weights_file)
    print('Successfully loaded weights')
  except ValueError as e:
    # If direct loading fails (likely due to layer mismatch from multi-GPU training),
    # try loading by layer names which will match compatible layers
    if 'layers' in str(e) or 'layer' in str(e).lower():
      print('Warning: Direct weight loading failed (likely multi-GPU saved weights).')
      print('Attempting to load weights by layer names...')
      try:
        model.load_weights(weights_file, by_name=True)
        print('Successfully loaded weights by layer names')
      except Exception as e2:
        print('Error: Could not load weights even by name.')
        print('Error details:', str(e2))
        raise e2
    else:
      # Re-raise if it's a different type of ValueError
      raise e

  test(model, model_input_dir_path, output_dir_npy, output_dir_png,
       pkl_file, test_file, weights_file, mgpu, has_manual_seg, export_png)
