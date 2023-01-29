import numpy as np
from scipy.ndimage import morphology


def surfd(input1, input2, sampling=1, connectivity=1):

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    # S=np.bitwise_xor(input_1 , morphology.binary_erosion(input_1, conn))
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)
    # Sprime=(input_2 , morphology.binary_erosion(input_2, conn))

    dta = morphology.distance_transform_edt(~S ,sampling)
    dtb = morphology.distance_transform_edt(~Sprime ,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])


    return sds


# In[10]:


def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`asd`
    :func:`hd`

    Notes
    -----
    This is a real metric, obtained by calling and averaging

    #>>> asd(result, reference)

    and

    #>>> asd(reference, result)

    The binary images can therefore be supplied in any order.
    """
    assd = np.mean( (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)) )
    return assd


# In[11]:


def asd(sds):
    """
    Average surface distance metric.

    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    input : surface distance metric
    -------
    asd : float
        The average surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing
        of elements along each dimension, which is usually given in mm.
   """
    asd = sds.mean()
    return asd


def msd(sds):
    """
    Average surface distance metric.

    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    input : surface distance metric
    -------
    asd : float
        The average surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing
        of elements along each dimension, which is usually given in mm.
   """
    asd = sds.max()
    return asd

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
        depth=segm.shape[2]
    except IndexError:
        raise

    return height, width,depth

def check_size(eval_segm, gt_segm):
    h_e, w_e,d_e = segm_size(eval_segm)
    h_g, w_g,d_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g) or (d_e!=d_g):
        raise EvalVolErr("DiffDim: Different dimensions of matrices!")

class EvalVolErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
