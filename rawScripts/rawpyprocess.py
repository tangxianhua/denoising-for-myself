import rawpy
import imageio
import rawpy.enhance#主要是坏点校正
from rawpy._rawpy import FBDDNoiseReductionMode, ColorSpace, HighlightMode
from rawpy._rawpy import DemosaicAlgorithm
import time
import os
path = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\origin\DSC00039.ARW'
savepath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\samgain\libraw'
savename = 'DSC00039librawLight'
starttime = time.time()
with rawpy.imread(path) as raw:
    rgb = raw.postprocess(
        output_bps=16,
        #去噪
        fbdd_noise_reduction = FBDDNoiseReductionMode.Light,
        noise_thr = None,  # threshold for wavelet denoising (default disabled)
        #demosaic
        demosaic_algorithm = DemosaicAlgorithm.AHD,
        four_color_rgb=False,  # whether to use separate interpolations for two green channels
        dcb_iterations=0,  # number of DCB correction passes, requires DCB demosaicing algorithm
        dcb_enhance=False,  # DCB interpolation with enhanced interpolated colors
        median_filter_passes=0,  # number of median filter passes after demosaicing to reduce color artifacts
        #wb
        use_auto_wb = True,
        use_camera_wb=False,  # whether to use the as-shot white balance values
        user_wb=None,  # list of length 4 with white balance multipliers for each color
        #blc
        user_black = None,  # custom black level
        #饱和度
        user_sat = None,#saturation adjustment (custom white level)
        #亮度
        no_auto_bright = True,#False  whether to disable automatic increase of brightness
        bright=2.0,  # brightness scaling
        auto_bright_thr=None,# ratio of clipped pixels when automatic brighness increase is used(see `no_auto_bright`). Default is 0.01 (1%).
        # 曝光
        exp_shift=None,# exposure shift in linear scale. Usable range from 0.25 (2-stop darken) to 8.0 (3-stop lighter).
        exp_preserve_highlights=0.0,# preserve highlights when lightening the image with `exp_shift`.From 0.0 to 1.0 (full preservation).
        #gamma
        gamma=(2.222, 4.5),# pair (power,slope), default is (2.222, 4.5) for rec. BT.709
        #色彩校正
        chromatic_aberration=None,  # pair (red_scale, blue_scale), default is (1,1),corrects chromatic aberration by scaling the red and blue channels
        #others
        no_auto_scale=False,  # Whether to disable pixel value scaling
        adjust_maximum_thr = 1,#0.75  see libraw docs
        highlight_mode = HighlightMode.Clip,
        # #others
        output_color = ColorSpace.sRGB,#output color space   raw,sRGB,Adobe,Wide,ProPhoto,XYZ
        half_size = False,
        user_flip = None, #0=none, 3=180, 5=90CCW, 6=90CW,default is to use image orientation from the RAW image if available
        bad_pixels_path = None)

endtime = time.time()
print('cost time',endtime-starttime)
rgb.tofile(os.path.join(savepath,savename+'.raw'))
#imageio.imsave(os.path.join(savepath,savename+'.tiff'), rgb)


