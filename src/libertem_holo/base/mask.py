import numpy as np
from skimage.draw import polygon
from skimage.draw import line
from scipy.ndimage import gaussian_filter

def line_filter(size, sidebandpos, width, length):
    """
    A line filter function that is used to remove Fresnel fringes from biprism.

    Parameters
    ----------
    size : 2d tuple, ()
        size of the FFT of the hologram.
    sidebandpos : 2d tuple, ()
        Position of the sideband that is used for reconstruction of holograms.
    width: float
        Width of the line (rectangle).
    length : float
        Length of the line (rectangle).
    Returns
    -------
        2d array containing line filter
    """

    angle = np.arctan2(size[0] / 2 + 1 - sidebandpos[0],  size[1] / 2 + 1 - sidebandpos[1])
    mid_pos = ((size[0] / 2 + 1 + sidebandpos[0]) / 2,
               (size[1] / 2 + 1 + sidebandpos[1]) / 2)
    left_bottom = (mid_pos[0] - length / 2 * np.cos(angle) + width / 2 * np.sin(angle),
                   (mid_pos[1] - length / 2 * np.sin(angle) - width / 2 * np.cos(angle)))
    right_bottom = (left_bottom[0] + np.cos(angle) * length,
                    left_bottom[1] + np.sin(angle) * length)
    left_top = (left_bottom[0] - np.sin(angle) * width,
                left_bottom[1] + np.cos(angle) * width)
    right_top = (right_bottom[0] + left_top[0] - left_bottom[0],
                right_bottom[1] + left_top[1] - left_bottom[1])
    r = np.array([left_bottom[0], right_bottom[0], right_top[0], left_top[0]], dtype=int)
    c = np.array([left_bottom[1], right_bottom[1], right_top[1], left_top[1]], dtype=int)
    rr, cc = polygon(r, c)
    mask = np.ones(size)
    mask[rr, cc] = 0

    return mask

def line_filter_gauss(size,sidebandpos,width,length,slice_fft,smoothness=1):
    """
    A line filter function that is used to remove Fresnel fringes from biprism.
    Parameters. The line will be created with skimage.draw.line. The starting points are
    the sideband position. The end points depend on the length and in the direction to top
    right image. 
    ----------
    size : 2d tuple, ()
        size of the FFT of the hologram.
    sidebandpos : 2d tuple, ()
        Position of the sideband that is used for reconstruction of holograms.
    width: pixel
        Width of the line (rectangle).
    length : pixel
        Length of the line (rectangle).
    Smoothnes : float
        Smoothness of the line. The value of sigma for gaussian filter.
    Returns
    -------
        2d array containing line filter
        
    """
    
    start_pos = (sidebandpos[0], sidebandpos[1])
    angle = np.arctan2(sidebandpos[0], size[1] - sidebandpos[1])
    end_pos = (sidebandpos[0] - np.int(np.floor(length * np.sin(angle))), sidebandpos[1] + np.int(np.floor(length * np.cos(angle))))
    
    rr,cc = line(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    mask = np.ones(size)
    mask[rr, cc] = 0
    
    for i in range(0,np.int(np.ceil(width/2))):
        rr,cc = line(start_pos[0], start_pos[1] + i, end_pos[0] + i, end_pos[1])
        mask[rr,cc] = 0 
        rr,cc = line(start_pos[0], start_pos[1] - i, end_pos[0], end_pos[1] - i)
        mask[rr,cc] = 0 
    
    mask = gaussian_filter(mask, sigma = smoothness)
    mask = np.fft.fftshift(np.fft.fftshift(mask)[slice_fft])
    
    return mask