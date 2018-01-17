import numpy as np
import random

def coTransform_flip(input_np_tuple, axes = (-2, -1), randomFlip = True):
    """
    flip the inputs in the tuple together. Can implement randomFlip.

    -------
    inputs:
        input_np_tuple: (input1, input2, ...)
        axes: flip the axis one by one
        randomFlip: if True, randomly drop some flip operations

    -------
    examples:
    >>> input_np = np.arange(12).reshape((3,4))
    >>> output_np_tuple = coTransform_flip(input_np_tuple = (input_np, input_np), axes = (-1, -2), randomFlip = False)
    >>> np.allclose(output_np_tuple[1], input_np[::-1, ::-1])
    True
    >>> output_np_tuple = coTransform_flip(input_np_tuple = (input_np, )*2, axes = (-1, -2), randomFlip = True)
    >>> np.allclose(output_np_tuple[0], output_np_tuple[1])
    True
    """

    if randomFlip:
        axes_randomFlip = random.sample(axes, random.randrange(len(axes)))
        axes = axes_randomFlip
    output_np_tuple = [None for _ in range(len(input_np_tuple))]
    for _i, _input_np in enumerate(input_np_tuple):
        dt = _input_np.dtype
        for _axis in axes:
            _input_np = np.flip(_input_np, axis = _axis)
        output_np_tuple[_i] = _input_np.astype(dt)
    return output_np_tuple


def coTransform_crop(input_np_tuple, output_shape = (-1, 8, 6), randomCrop = True):
    """
    crop patches from the inputs tuple together

    -------
    inputs:
        input_np_tuple: (input1, input2, ...), they need to have the same size for the axis 2B cropped
        output_shape: output shape. 
                If -1, keep the current size of the axis. 
                Make sure not larger than the input shape.
        randomFlip: If True, randomly crop patches with shape = output_shape
                If False, after crop,  the center don't change

    -------
    examples:
    >>> input1_np = np.arange(36).reshape((1, 6, 6))
    >>> input2_np = np.arange(72).reshape((2, 6, 6))
    >>> output_np_tuple = coTransform_crop(input_np_tuple = (input1_np, input2_np), output_shape = (1, 6, 2), randomCrop = False)
    >>> np.allclose(output_np_tuple[0], input1_np[:, :, 2:4])
    True
    >>> np.allclose(output_np_tuple[1], input2_np[0, :, 2:4])
    True
    >>> output_np_tuple = coTransform_crop(input_np_tuple = (input1_np, input2_np), output_shape = (-1, 8, 2), randomCrop = False)
    Warning: the output shape (-1, 8, 2) should not larger than the input shape (1, 6, 6)
    >>> np.allclose(output_np_tuple[0], input1_np[:, :, 2:4])
    True
    >>> np.allclose(output_np_tuple[1], input2_np[:, :, 2:4])
    True
    >>> input_np = np.arange(24).reshape((2, 3, 4))     # test randomCrop
    >>> output_np_tuple = coTransform_crop(input_np_tuple = (input_np, )*2, output_shape = (1, 2, 2), randomCrop = True)
    >>> np.allclose(output_np_tuple[0], output_np_tuple[1])
    True
    """

    nDim = len(output_shape)
    output_np_tuple = [None for _ in range(len(input_np_tuple))]

    # define the shared accessing slice
    _slc = [slice(None) for _ in range(nDim)]   # by default the slice will access all the content
    _input_np = input_np_tuple[0]
    _input_input_axisShape = _input_np.shape
    for _axis, _input_axisShape in enumerate(_input_input_axisShape):
        patch_axisShape = output_shape[_axis]
        if _input_axisShape < patch_axisShape:
            print("Warning: the output shape {} should not larger than the input shape {}".format(output_shape, _input_input_axisShape))
            continue

        if (patch_axisShape == -1):
            continue   # by default the slice will access all the content
        else:
            patch_radius = patch_axisShape / 2
            odd_axisShape = patch_axisShape % 2     # if the output axisShape is odd, need to add 1 to the endIndex of the slice
            center_pixel = (patch_radius + np.random.randint(0, _input_axisShape - patch_axisShape + 1)) if randomCrop \
                    else _input_axisShape / 2
            _slc[_axis] = slice(center_pixel - patch_radius, center_pixel + patch_radius + odd_axisShape)

    
    output_np_tuple = tuple([_input_np[_slc] for _input_np in input_np_tuple])
    return output_np_tuple



import doctest
doctest.testmod(optionflags=doctest.ELLIPSIS)
