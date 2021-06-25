# Center Smoothing

This is the code repository for the paper titled Center Smoothing: Provable Robustness for Functions with Metric-Space Outputs.
The main code for Center Smoothing procedure is available in the file center_smoothing.py.
The model architectures are available in models.py.
Various distance functions used in the experiments can be found in distance_functions.py.


To generate certificates, run:

```
python facenet-celeba.py [path to CelebA dataset] [path to save certificates] --eps_in [value of epsilon in] --sigma [value of sigma]
python test_dim_red.py [path to trained model] [distance_function] [path to save certificates] [dataset] --eps_in [value of epsilon in] --sigma [value of sigma] --latent_dim [number of latent dimensions]
python test_reconstructor.py [path to trained model] [distance_function] [path to save certificates] [dataset] [path to measurement matrix] --eps_in [value of epsilon in] --sigma [value of sigma]
python gan_lpips.py [path to save certificates] --eps_in [value of epsilon in] --sigma [value of sigma]
```


To compute median certificate, run:

```
python get_median_cert.py [path to certificate file]
```


To train models, run:

```
python train_dim_red.py [path to save trained model] [dataset] --latent_dim [number of latent dimensions] --sigma [value of sigma]
python train_reconstructor.py [path to save trained model] [dataset] [path to measurement matrix] --sigma [value of sigma]
```
