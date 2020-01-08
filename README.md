# Rust - Monocular Depth Prediction

This is the reference Rust LibTorch implementation for testing depth estimation models using the method described in:

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
>
> [ICCV 2019](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>


Mono RGB | Depth Prediction
------------ | -------------
![](assets/test_image_01.png) | ![](assets/test_depth_01.jpg)
![](assets/test_image_02.png) | ![](assets/test_depth_02.jpg)
![](assets/test_image_03.png) | ![](assets/test_depth_03.jpg)
![](assets/test_image_04.png) | ![](assets/test_depth_04.jpg)


```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```