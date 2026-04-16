import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from scipy.interpolate import RBFInterpolator
from matplotlib.colors import SymLogNorm

import functions_beams as fbeams


#Useful shortcuts: 
# CTRL + K,0  --->  collapse all functions
# CTRL + K,J  --->  expand all functions

#############################
### Build rotation matrix ####   preserving image orientation  
#############################

def rotmatrix_frame_vec(v1, v2):  #rotation matrix -- inputs need to be unit vectors
    '''
    Build rotation matrix from vectors to beam centeres --> use for beam rotation tests 
    Args: v1c, v2c = unit vectors to beam center and destination position
    Return: R = rotation matrix 
    '''
    z_axis = np.array([0., 0., 1.])

    z1 = v1 # / np.linalg.norm(v1) -- no need to normalize here because v1 and v2 are already unit vectors
    x1 = np.cross(z_axis, z1)
    x1_norm_sq = x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2]
    if x1_norm_sq < 1e-16:
        x1 = np.array([1., 0., 0.])  # Handle polar case
        x1_norm_sq=1.
    x1 *= 1./np.sqrt(x1_norm_sq)
    y1 = np.cross(z1, x1)
    R1 = np.column_stack([x1, y1, z1])  # Columns are the local basis vectors


    z2 = v2 #/ np.linalg.norm(v2)
    x2 = np.cross(z_axis, z2)
    x2_norm_sq = x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2]
    if x2_norm_sq < 1e-16:
        x2 = np.array([1., 0., 0.])
        x2_norm_sq = 1.
    x2 *= 1./np.sqrt(x2_norm_sq)
    y2 = np.cross(z2, x2)
    R2 = np.column_stack([x2, y2, z2])

    R = R2 @ R1.T

    return R


def rotmatrix_frame(R1, v2):  #rotation matrix preservinf the image orientation -- inputs need to be unit vectors
    '''
    Build rotation matrix to beam destination --> use for convolution
    Args: 
    - R1 = rotation matrix to original beam position
    - v2c = unit vector to destination position
    Return: R = rotation matrix 
    '''
    z_axis = np.array([0., 0., 1.])

    z2 = v2 #/ np.linalg.norm(v2)
    x2 = np.cross(z_axis, z2)
    x2_norm_sq = x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2]
    if x2_norm_sq < 1e-16:
        x2 = np.array([1., 0., 0.])
        x2_norm_sq = 1.
    x2 *= 1./np.sqrt(x2_norm_sq)
    y2 = np.cross(z2, x2)
    R2 = np.column_stack([x2, y2, z2])

    R = R2 @ R1.T

    return R


def rotmatrix_frame_opt(R1, v2):  # checked that this is faster than many other forms
    '''
    Build rotation matrix to beam destination: optimized version for final run
    Args: 
    - R1 = rotation matrix to original beam position
    - v2c = unit vector to destination position
    Return: R = rotation matrix 
    '''
    z2 = v2
    
    # # Manual cross product instead of np.cross (faster for 3D vectors) -- Since z_axis = [0,0,1], this simplifies to:
    # x2 = [-z2[1], z2[0], 0]
    x2_0 = -z2[1]
    x2_1 = z2[0]
    #x2_2 = 0.0
    
    # Manual norm squared calculation
    x2_norm_sq = x2_0*x2_0 + x2_1*x2_1 #+ x2_2*x2_2
    
    if x2_norm_sq < 1e-16:
        x2_0, x2_1 = 1.0, 0.0  #x2_0, x2_1, x2_2 = 1.0, 0.0, 0.0
        x2_norm_sq = 1.0
    
    # Manual normalization (avoiding sqrt when possible)
    inv_norm = 1.0 / np.sqrt(x2_norm_sq)
    x2_0 *= inv_norm
    x2_1 *= inv_norm
    #x2_2 *= inv_norm
    
    # Manual cross product for y2 = np.cross(z2, x2)
    y2_0 = - z2[2] * x2_1 #+ z2[1] * x2_2 
    y2_1 = z2[2] * x2_0 #- z2[0] * x2_2  
    y2_2 = z2[0] * x2_1 - z2[1] * x2_0
    
    # Manual matrix construction and multiplication R2 @ R1.T
    # R2 = [[x2_0, y2_0, z2[0]],
    #       [x2_1, y2_1, z2[1]], 
    #       [x2_2, y2_2, z2[2]]]
    # R = R2 @ R1.T

    R = np.empty((3, 3))  # Pre-allocate result matrix
    
    # Manual matrix multiplication 
    R[0, 0] = x2_0 * R1[0, 0] + y2_0 * R1[0, 1] + z2[0] * R1[0, 2]
    R[0, 1] = x2_0 * R1[1, 0] + y2_0 * R1[1, 1] + z2[0] * R1[1, 2]
    R[0, 2] = x2_0 * R1[2, 0] + y2_0 * R1[2, 1] + z2[0] * R1[2, 2]
    
    R[1, 0] = x2_1 * R1[0, 0] + y2_1 * R1[0, 1] + z2[1] * R1[0, 2]
    R[1, 1] = x2_1 * R1[1, 0] + y2_1 * R1[1, 1] + z2[1] * R1[1, 2]
    R[1, 2] = x2_1 * R1[2, 0] + y2_1 * R1[2, 1] + z2[1] * R1[2, 2]
    
    R[2, 0] = y2_2 * R1[0, 1] + z2[2] * R1[0, 2]
    R[2, 1] = y2_2 * R1[1, 1] + z2[2] * R1[1, 2]
    R[2, 2] = y2_2 * R1[2, 1] + z2[2] * R1[2, 2]

    
    return R


def get_R1(v1): 
    '''
    Build first half of the rotation matrix
    Args: v1 = vector to original beam center
    '''
    z_axis = np.array([0., 0., 1.])

    z1 = v1 # / np.linalg.norm(vc1) -- no need to normalize here because v1 and v2 are already unit vectors
    x1 = np.cross(z_axis, z1)
    x1_norm_sq = x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2]
    if x1_norm_sq < 1e-16:
        x1 = np.array([1., 0., 0.])  # Handle polar case
        x1_norm_sq=1.
    x1 *= 1./np.sqrt(x1_norm_sq)
    y1 = np.cross(z1, x1)
    R1 = np.column_stack([x1, y1, z1])  # Columns are the local basis vectors

    return R1



#######################
### Beam rotation  ####     
#######################

def beam_grid(map_beam, v1c, angdistR): 
    '''
    Get quantities to create an interpolation grid at beam origin
    Args: beam map, beam center, beam radius 
    Return: 
    - v1: original beam pixel centers potision 
    - beam1: beam values at those positions
    '''
    nside=hp.get_nside(map_beam)

    beam_idpix1 = hp.query_disc(nside=nside, vec=v1c, radius=angdistR, inclusive=False, nest=False, buff=None)
    print(len(beam_idpix1), 'pixels used for interpolation')

    beam_idpix0 = np.where(map_beam != 0)  
    beam_idpix0 = np.array(beam_idpix0[0])
    if len(beam_idpix1)<len(beam_idpix0): 
        print('ERROR: not all beam pixels are sampled')
    v1=hp.pix2vec(nside, beam_idpix1)

    v1=np.array(v1) # v1: interpolation grid points at original beam position (position of pixels inside angdistR)
    beam1=map_beam[beam_idpix1] #beam1: corresponding original beam values: interpolation grid values

    return v1, beam1 

#from v1c,v2c --> use for rotation tests
def beamrotation_test(v1c, v2c, v1, beam0, nside, angradius):  #need v1c, v2c to define the rotation --> rotate all vectors to destination --> interpolate the beam there
    '''
    rotate beam from v1c to v2c
    APPROACH 1: rotate original beam to destination - interpolate at destination
    interpolation grid built from: positions of original beam pixels rotated to destination position + original beam values
    Ags: 
    - v1c, v2c: pointing at beam center initial and destination --> define the rotation matrix R
    - v1, beam0: vectors to initial pixels + corresponding beam values 
    - angradius: radius to select destination pixels (take it as x% larger than initial beam redius) - in radias
    Return: new beam map (healpix array)
    '''
    R=rotmatrix_frame_vec(v1c, v2c)
    v2 =hp.rotator.rotateVector(R, v1, do_rot=True) #initial pixels rotated to destination = interpolation grid at destination (values given by beam0)

    beam_idpix2_interp=hp.query_disc(nside=nside, vec=v2c, radius=angradius, inclusive=False, nest=False, buff=None)  # get indeces of pixels at destination
    v2_interp=np.array(hp.pix2vec(nside, beam_idpix2_interp)) #get corresponding vectors

    interpolator = RBFInterpolator(v2.T, beam0, kernel='cubic')  #build interpolation grid at destination // options: 'thin_plate_spline' or 'multiquadric', 'cubic' (linear and quintic don't work well) 
    beam_interp = interpolator(v2_interp.T) #interpolate to destination pixels positions

    npix=12*nside**2 #build rotated beam map
    beam_map2=np.zeros(npix)
    beam_map2[beam_idpix2_interp]=beam_interp
 
    return beam_map2  


def beamrotation_test_opt(v1c, v2c_i, interpolator0, nside, angradius):
    '''
    rotate beam from v1c to v2c - optimized for full run [build interpolator just once]
    #APPROACH 2: rotate destination pixels positions to origin - interpolate at the origin
    #interpolation grid built from: positions of original beam pixels + original beam values
    Ags:
    - v1c, v2c: pointing at beam center initial and destination --> define the rotation matrix R
    - interpolator0: contains beam values at initial pixels positions
    - angradius: radius to select destination pixels (take it as x% larger than initial beam redius) - in radias
    Return: 
    - new beam mapvectors
    '''
    R=rotmatrix_frame_vec(v1c, v2c_i)
   
    beam_idpix2_interp = hp.query_disc(nside=nside, vec=v2c_i, radius=angradius,inclusive=False, nest=False, buff=None)  # get indeces of pixels at destination
    v2_interp = np.array(hp.pix2vec(nside, beam_idpix2_interp)) #get corresponding vectors
    
    R_inv = R.T  # inverse rotation (transpose for orthogonal matrix)
    v1_interp = R_inv @ v2_interp #rotate destination pixels positions to original beam position
    #v1_interp = hp.rotator.rotateVector(R_inv, v2_interp, do_rot=True)  #<--- alternative

    beam_interp = interpolator0(v1_interp.T)  #interpolate to destination pixels rotated to original position (interpolation grid build at initial position)

    npix=12*nside**2 #build rotated beam map
    beam_map2=np.zeros(npix)
    beam_map2[beam_idpix2_interp]=beam_interp
    
    return beam_map2


def beamrotation_test_opt_fast(R1, v2c_i, interpolator0, nside, angradius):
    '''
    rotate beam from v1c to v2c - optimized for full run [build interpolator just once]
    #APPROACH 2: rotate destination pixels positions to origin - interpolate at the origin
    #interpolation grid built from: positions of original beam pixels + original beam values
    Ags:
    - R1: fist half of the rotation matrix, defined from v1c
    - v2c: pointing at destination
    - interpolator0: contains beam values at initial pixels positions
    - angradius: radius to select destination pixels (take it as x% larger than initial beam redius) - in radias
    Return: 
    - new beam mapvectors
    '''
    R=rotmatrix_frame_opt(R1, v2c_i)  #most optimized rotation matrix building
   
    beam_idpix2_interp = hp.query_disc(nside=nside, vec=v2c_i, radius=angradius,inclusive=False, nest=False, buff=None)  # get indeces of pixels at destination
    v2_interp = np.array(hp.pix2vec(nside, beam_idpix2_interp)) #get corresponding vectors
    
    v1_interp = R.T @ v2_interp   # inverse rotation: transpose for orthogonal matrix

    beam_interp = interpolator0(v1_interp.T)  #interpolate to destination pixels rotated to original position (interpolation grid build at initial position)

    npix=12*nside**2 #build rotated beam map
    beam_map2=np.zeros(npix)
    beam_map2[beam_idpix2_interp]=beam_interp
    
    return beam_map2


#########################
### Beam convolution ####     
#########################

# convolution in 1 spot ---> loop over all pixels 
def convolve_1pix(R1, v2c_i, interpolator0, nside, angradius, pixArea, cmb_map ):  #NOTA: pixel area has to be in radians!!
    '''
    Args [same as beam rotation]: 
    - R1: first half of the rotation matrix, defined from v1c
    - v2c_i: vector to destination pixel (where to convolve)
    - interpolator0: contains beam values at initial pixels positions (from original beam map)
    - nside: of the maps
    - angradius: radius to select destination pixels (take it as x% larger than initial beam redius) - in radias
    Args[new for convolution]:
    - pixArea: pixel area in radians
    - cmb_map: map to convolve (CMB map)
    Return: 
    - T_convolved_i: convolved T value at destination pixel v2c_i
    '''
    beam_idpix2_interp=hp.query_disc(nside=nside, vec=v2c_i, radius=angradius, inclusive=False, nest=False, buff=None)  # get indeces of pixels around v2c
    v2_interp = np.vstack(hp.pix2vec(nside, beam_idpix2_interp)) # vectors pointing at destination pixels

    R=rotmatrix_frame_opt(R1, v2c_i)     
    v1_interp = R.T @ v2_interp  # v1_interp = np.dot(R_inv, v2_interp)  # vectors pointing at corresponding directions at original beam position
 
    beam_interp = interpolator0(v1_interp.T) # Use the pre-computed interpolator with the rotated query points
    cmb_values=cmb_map[beam_idpix2_interp]
    #T_convolved_i = sum(cmb_map[beam_idpix2_interp]*beam_interp)*pixArea
    T_convolved_i = np.dot(cmb_values, beam_interp) * pixArea 
    return  T_convolved_i

#faster version, obtained by condensing operations
#checked that it produces the same results as the other one
def convolve_1pix_contracted(R1, v2c_i, interpolator0, nside, angradius, pixArea, cmb_map ):   #NOTA: pixel area has to be in radians!!
    beam_idpix2_interp=hp.query_disc(nside=nside, vec=v2c_i, radius=angradius, inclusive=False, nest=False, buff=None)
    v1_interp = rotmatrix_frame_opt(R1, v2c_i).T @ np.vstack(hp.pix2vec(nside, beam_idpix2_interp))  
    return np.dot(cmb_map[beam_idpix2_interp], interpolator0(v1_interp.T)) * pixArea 

# set gnomview parameters to show image properly
def set_gnomeview(angsize_img, xside, nside): 
    '''
    Args: 
    angsize_img: angukar size of gnomeview image in degrees
    xside: number of pixels per side of the image [xsize parameter]
    --> computes the [reso] parameter: size of each pixel in the gnamview image, from eq: ang_size(deg)=reso*xsize/60
    '''
    reso=angsize_img*60/xside  #compute pix dimension

    pixArea_deg= hp.nside2pixarea(nside, degrees=True) 
    pixsize_map=np.sqrt(pixArea_deg)
    print('image pixels size = ', reso/pixsize_map, 'times initial pixels')
    return reso

# get quantities needed to perform convolution on a disk 
def get_convolution_quantities(map_cmb, map_beam, v1c, v2c, angradius_conv):
    '''
    Args: 
    - map_cmb: map to convolve
    - map_beam: beam map
    - v1c: vector to original beam center (of beam in map_beam)
    - v2c: vector to destination position (center of disk to convolve)
    - angradius_conv: radius of the disk to convolve, in degrees
    Return: 
    - quantities for Args of convolve_1pix: R1, interpolator0, nside, pixArea_rad, angdistR-->angradius
    - idpix_disk: indeces of pixels in the disk to convolve (to loop over for convolution --> get v2c_i)
    '''
    #get Nside
    nside1=hp.get_nside(map_cmb) 
    nside2=hp.get_nside(map_beam)
    if nside1!=nside2: 
        print('\nNside of beam map and CMB realizations do not coincide')
        nside=min(nside1, nside2)
        print('downgrading to lower resolution map: nside = ', nside )
        map_cmb=hp.pixelfunc.ud_grade(map_cmb, nside)
        map_beam_ini=hp.pixelfunc.ud_grade(map_beam_ini, nside)
    elif nside1==nside2: 
        nside=nside1 #if they are the same, set common nside
        print('\nchecked that the two maps have the same Nside=', nside)

    #get rotation quantities:
    R1=get_R1(v1c)
    angdistR=fbeams.beam_radius(map_beam, v1c)
    print('\nbeam dimension (rad) = ', angdistR )
    v1, beam1 = beam_grid(map_beam, v1c, angdistR)
    interpolator0 = RBFInterpolator(v1.T, beam1, kernel='cubic')

    #get rotation quantities:
    R1=get_R1(v1c)
    angdistR=fbeams.beam_radius(map_beam, v1c)
    print('\nbeam dimension (rad) = ', angdistR )
    v1, beam1 = beam_grid(map_beam, v1c, angdistR)
    interpolator0 = RBFInterpolator(v1.T, beam1, kernel='cubic')

    ## get quantities for convolution
    pixArea_rad= hp.nside2pixarea(nside, degrees=False) 

    idpix_disk=hp.query_disc(nside=nside, vec=v2c, radius=np.deg2rad(angradius_conv), inclusive=False, nest=False, buff=None)

    return nside, R1, angdistR, interpolator0, pixArea_rad, idpix_disk


# compare areas (diks) of 2 maps --> plot maps and histograms of difference, relative difference 
def compare_map_area(map1, map2, vc, disk_angradius , scale='hist', titles=['Original T field','Convolved T field']):  
    '''
    Args: 
    - map1, map2: maps to compare
    - vc, disk_angradius: center and radius of the disk area to compare (in degrees)
    - scale: 'hist' or 'linear' --> scale for gnomview images 
    - titles: list of titles to describe the two maps to compare.
    Plots: 
    - maps: map1, map2, difference map (or relative difference map)
    - histograms: of values from difference and relative difference maps (cut at a certain percentile to show the bulk of the distribution)
    '''

    fig = plt.figure(figsize=(12, 5.5))

    angc=hp.pixelfunc.vec2ang(np.array(vc), lonlat=True)
    phic=angc[0][0]
    thetac=angc[1][0]
    print(f'\nmaps centered at (phi, theta) = ({phic},{thetac})')

    map2_masked = np.where(map2==0, hp.UNSEEN, map2)
    map_diff = np.where(map2_masked != hp.UNSEEN, map1 - map2, hp.UNSEEN)
    map1_circle = np.where(map2_masked != hp.UNSEEN, map1, hp.UNSEEN)
    map_diff_rel = np.where(map2_masked != hp.UNSEEN, (map1 - map2)/map1, hp.UNSEEN)

    valid1 = map1_circle != hp.UNSEEN
    valid_diff = map_diff != hp.UNSEEN

    print(f'\n(min,max) original T field: ({np.min(map1_circle[valid1]):.2f} , {np.max(map1_circle[valid1]):.2f})')
    print(f'(min,max) convolved T field: ({np.min(map2):.2f} , {np.max(map2):.2f})')
    print(f'(min,max) difference: ({np.min(map_diff[valid_diff]):.2f} , {np.max(map_diff[valid_diff]):.2f})')
    print(f'(min,max) relative difference: ({np.min(map_diff_rel[valid_diff]):.2f} , {np.max(map_diff_rel[valid_diff]):.2f})')


    # PLOT maps
    xsize=1500
    nside=hp.get_nside(map1)
    reso=set_gnomeview(2.2*disk_angradius, xsize, nside)
    hp.visufunc.gnomview(map1_circle, rot=[phic, thetac], reso=reso, xsize=xsize, norm=scale, title=titles[0], return_projected_map=True , sub=(1, 3, 1)) #, min=-300, max=300)  
    hp.visufunc.gnomview(map2_masked, rot=[phic, thetac], reso=reso, xsize=xsize, norm=scale, title=titles[1], return_projected_map=True , sub=(1, 3, 2)) #, min=-300, max=300)  
    hp.visufunc.gnomview(map_diff, rot=[phic, thetac], reso=reso, xsize=xsize, norm=scale, title='Difference', return_projected_map=True , sub=(1, 3, 3)) #, min=-300, max=300)  
    plt.show()

    #PLOT histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(map_diff[valid_diff], bins=100, color='steelblue')
    ax1.set_title('Difference')

    percentile_=96
    arr = map_diff_rel[valid_diff]
    threshold = np.percentile(np.abs(arr), percentile_)
    filtered_data = arr[np.abs(arr) <= threshold]

    print(r'cut realtive diff hist to show '+str(percentile_)+r'% of data points: threshold =', threshold)
    ax2.hist(filtered_data, bins=100, color='darkblue')
    ax2.set_title('Relative difference (cut at '+str(percentile_)+r' percentile)')

    if threshold<1e-2:
        ax2.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()
    
    return 1


## function to perform convolution of disk and show result: buits from previous functions
def convolve_disk(map_cmb, map_beam, v1c, phic2, thetac2, angradius_conv):  #angradius_conv to be given in degrees
    '''
    Args: 
    - map_cmb: map to convolve
    - map_beam, v1c: beam map and vector to original beam center
    - phic2, thetac2: coordinates of destination position (center of disk to convolve) in degrees
    - angradius_conv: radius of the disk to convolve, in degrees
    Return: 
    - map_conv: convolved map
    '''

    #get quantities for beam rotation and convolution
    v2c=hp.pixelfunc.ang2vec(phic2, thetac2, lonlat=True)
    nside, R1, angdistR, interpolator0, pixArea_rad, idpix_disk = get_convolution_quantities(map_cmb, map_beam, v1c, v2c, angradius_conv)

    x,y,z = hp.pixelfunc.pix2vec(nside, ipix=idpix_disk) #get v2c for all other pixels in idpix_disk (check that array had the right shape)
    v_disk = np.vstack((x, y, z)).T 

    npix_disk=len(idpix_disk)
    print('\nnº of pixels to convolve:', len(idpix_disk))

    # perform convolution
    print('start convolution...')
    T_convolved=np.zeros(npix_disk)
    for i in range(len(idpix_disk)):
        T_convolved[i]=convolve_1pix(R1, v_disk[i], interpolator0, nside, angdistR, pixArea_rad, map_cmb ) 
    print('convolution ended')

    #create convolved map
    npix=12*nside**2 #get beam map 2
    map_conv=np.zeros(npix)
    map_conv[idpix_disk]=T_convolved

    #show results: 
    compare_map_area(map_cmb, map_conv, v2c, angradius_conv)

    return map_conv


