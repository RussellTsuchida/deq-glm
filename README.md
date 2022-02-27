# DECLARATIVE NETS THAT ARE EQUILIBRIUM MODELS

0. > pip install -r requirements.txt

# Fully connected architecture experiments
1. Make sure the directory pointed to by line 28 of code/scripts/regression_test.py exists
2. ``` python -um code.scripts.regression_test <SEED> ```
  where <SEED> is an integer representing the random seed. Run this multiple times for multiple random seeds.
3. ```python -um code.lib.hist_plot```
# Convolutional architecture experiments
1. (optional) If you would like to use the HSI dataset, download and extract it using the instructions at https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road. That is, ``` wget https://hsiroad-sh.oss-cn-shanghai.aliyuncs.com/hsi_road.tar.gz``` and then ```tar -xvf hsi_road.tar.gz```. Delete the images/*_rgb.tiff files.
2. Open code/scripts/denoise_test.py. Change the number of channels on line 30 to whatever you wish. Change line 31 to 'hsi' or 'cifar' depending on which dataset you would like to use. Make sure the directory pointed to by lin 21 exists --- this is where the experiment output data is stored. 
3. ```python -um code.scripts.denoise_test <SEED>``` where <SEED> is an integer representing the random seed. This will do a full spectral norm sweep for a single seed. Repeat as many times as you wish.
4. To plot the results, point to the experiment output data directory in line 116 of code/lib/hist_plot_cifar.py. Then run 
  ```python -um code.lib.hist_plot_cifar```
# Kernel ridge regression == DEQ test (Appendix G)
1. ```python -um code.scripts.mnist''' will run through the full MNIST dataset and plot images that are different as .pdf files. It will also produce the plot in Figure 14.
  
