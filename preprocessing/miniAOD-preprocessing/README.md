# preprocessing of CNN data
- edit the script `preprocessing_cnn_map.py` to your preferences:
    - adjust pixelwidths and eta/phi ranges

- look into `preprocessing_1_convert_miniAOD_to_h5.py` to adjust paths for miniAOD files and samples and outputs
- execute `preprocessing_1_convert_miniAOD_to_h5.py`:
    - submits one job per miniAOD file to the NAF batch system
    - each job creates an hdf5 file containing the CNN maps
    - afterwards the single files are added together
