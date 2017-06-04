import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib

# reference: http://nipy.org/nibabel/gettingstarted.html
# open an image
example_filename = os.path.join(data_path, 'example4d.nii.gz')
img = nib.load(example_filename)
print img.shape

# get numpy array
data = img.get_data()
print data.shape

# save an Image
data = np.zeros((32, 32, 15), dtype = np.int16)
data[10:20, 10:20, 5:10] = 1
img = nib.Nifti1Image(data, np.eye(4))
save_name = os.path.join("/Users/guotaiwang/Downloads", "test_img.nii.gz")
nib.save(img, save_name)