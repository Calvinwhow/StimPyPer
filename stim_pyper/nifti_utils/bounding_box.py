import os 
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output

class NiftiBoundingBox:
    def __init__(self, nifti_paths):
        """
        Initialize the bounding box utility with NIfTI file paths.
        Args:
            nifti_paths (list of str): List of file paths to NIfTI images.
        Attributes:
            self._stacked_data (numpy.ndarray or None): Stacked 3D array of NIfTI image data.
            self._collapsed_data (numpy.ndarray or None): Collapsed representation of the stacked NIfTI data.
        """
        self.nifti_paths = nifti_paths
        self._bounding_box_affine = None
        self._bounding_box_shape = None
        self._stacked_data = None
        self._voxel_size = None
        self.force_reslice = False
        self.sign = None

    @property
    def bounding_box_affine(self):
        return self._bounding_box_affine

    @property
    def bounding_box_shape(self):
        return self._bounding_box_shape

    @property
    def stacked_nifti(self):
        return self._stacked_data

    @property
    def voxel_size(self):
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size):
        if not isinstance(voxel_size, (tuple, list, np.ndarray)) or len(voxel_size) != 3:
            raise ValueError("voxel_size must be a 3-element tuple, list, or array.")
        self._voxel_size = tuple(voxel_size)
    
    # Internal helper methods
    def _load_nii(self, path):
        nii = nib.load(path)
        if self.force_reslice:
            nii = resample_to_output(nii, voxel_sizes=self.voxel_size, mode='nearest', order=0)
        return nii
    
    def _get_corners(self, nii):
        shape = nii.shape
        affine = nii.affine
        corner_origin = affine[:3, 3]
        corner_end = (affine @ np.array([shape[0], shape[1], shape[2], 1]))[:3]
        return corner_origin, corner_end

    def _generate_affine(self, min_coords):
        affine = np.eye(4)
        affine[:3, :3] *= self.voxel_size # set zooms
        affine[0, 0] *= self.sign # set orientation
        affine[:3, 3] = min_coords
        return affine

    def _generate_shape(self, min_coords, max_coords):
        shape = np.ceil((max_coords - min_coords) / self.voxel_size).astype(int)
        return tuple(np.abs(shape))
    
    def _generate_corner_coordinates(self):
        '''Generate the coordinates of the bounding box from the NIfTI files.'''
        origin_coords = []
        end_coords = []
        for path in self.nifti_paths:
            nii = self._load_nii(path)
            nii_origin, nii_end = self._get_corners(nii)
            origin_coords.append(nii_origin)
            end_coords.append(nii_end)
        return origin_coords, end_coords
    
    def _check_signs(self):
        '''Gather orientations and assess if reslicing is needed.'''
        sign_list = []
        for path in self.nifti_paths:
            nii = nib.load(path)
            sign_list.append( np.sign(np.diag(nii.affine[:3,:3])) )
        if not np.all(np.all(np.array(sign_list) == np.array(sign_list)[0], axis=1)):
            raise ValueError("Not all NIfTI files have the same orientation (sign of axes differs). Please reorient your images.")
        self.sign = np.sign(np.array(sign_list)[0][0])
    
    def _check_zooms(self):
        '''Gather zooms and to assess if reslicing is needed.'''
        zoom_list = []
        for path in self.nifti_paths:
            nii = nib.load(path)
            zoom_list.append(nii.header.get_zooms()[:3])
        zoom_array = np.array(zoom_list)  # shape: (n_files, 3)

        # If they’re not all identical, enforce isotropic reslicing:
        if not np.allclose(zoom_array.min(axis=0), zoom_array.max(axis=0)):
            max_dim = zoom_array.max()  # The largest dimension among any of the axes in any file to prevent aliasing
            self.voxel_size = (max_dim, max_dim, max_dim)
            self.force_reslice = True
            print(f"Multiple different voxel sizes detected. Using isotropic voxel size = {self.voxel_size} (lowest resolution)")
        else:
            self.voxel_size = tuple(zoom_array[0])
            print(f"All files appear to have the same voxel sizes: {self.voxel_size}. No reslicing needed.")
            
    def _pick_xform_code(self):
        """Return a single code to use for both qform/sform."""
        codes = [nib.load(p).header['sform_code'] for p in self.nifti_paths]
        codes = [int(c) for c in codes if c > 0] # keep only non-zero
        if not codes:
            return 1                       # nothing set → treat as scanner anat
        if len(set(codes)) == 1:
            return codes[0]                # all agree → preserve it
        return 1                           # mixed: choose the *least specific* space (1) to avoid lying

    # Public functions
    def generate_bounding_box(self):
        '''Creates the bounding box encompassing the largest offset and farthest point from the smallest offset'''
        self._check_zooms()
        self._check_signs()
        origin_coords, end_coords = self._generate_corner_coordinates()

        ultimate_origin = np.min(np.vstack(origin_coords), axis=0)
        ultimate_end = np.max(np.vstack(end_coords), axis=0)
        self._bounding_box_affine = self._generate_affine(ultimate_origin)
        self._bounding_box_shape = self._generate_shape(ultimate_origin, ultimate_end)

    def add_niftis_to_bounding_box(self):
        '''Find where the niftis sit in the bbox and put em there'''
        if self._bounding_box_affine is None or self._bounding_box_shape is None:
            raise ValueError("Bounding box not generated yet.")

        stacked_data = np.zeros(self._bounding_box_shape + (len(self.nifti_paths),), dtype=np.float32)

        for idx, path in enumerate(self.nifti_paths):
            nii = self._load_nii(path)
            offset_vox = np.round(np.linalg.inv(self._bounding_box_affine) @ nii.affine)[:3, 3].astype(int)
            nii_data = nii.get_fdata()
            shape = nii_data.shape
            x, y, z = offset_vox
            stacked_data[x:x+shape[0], y:y+shape[1], z:z+shape[2], idx] = nii_data

        self._stacked_data = stacked_data

    def collapse_bbox_to_3d(self):
        '''Collapse the stacked NIfTI data into a 3D tensor by summing along the fourth dimension.'''
        if self._stacked_data is None:
            raise ValueError("Stacked nifti data not generated yet.")
        
        self._collapsed_data = np.sum(self._stacked_data, axis=3)
        
    def collapsed_bbox_to_mask(self):
        '''Collapse the stacked NIfTI data into a 3D mask by thresholding.'''
        if self._collapsed_data is None:
            raise ValueError("collapse_bbox_to_3d not yet called.")
        return np.where(self._collapsed_data > 0, 1, 0)

    def save_nifti(self, data, output_path):
        '''Save the nifti to a location of your choice. Call self._stacked_data to save the 4D nifti'''
        if self._stacked_data is None:
            raise ValueError("Stacked nifti data not generated yet.")
        
        code = self._pick_xform_code()
        header = nib.Nifti1Header()
        header.set_xyzt_units('mm')
        header.set_sform(self._bounding_box_affine, code=code)
        header.set_qform(self._bounding_box_affine, code=code)
        header.set_data_dtype(np.float32)
        header['descrip'] = 'Stacked NIfTI with spatially aligned volumes'
        stacked_img = nib.Nifti1Image(data, affine=self._bounding_box_affine, header=header)
        nib.save(stacked_img, output_path)
        
    def run(self, output_dir):
        """
        Orchestrates bounding box generation, stacking, collapsing, and saving.
        
        params:
            output_path (str) : full path of directory to save the resultant nifti in. Must exist. 
        """
        self.generate_bounding_box()
        self.add_niftis_to_bounding_box()
        self.collapse_bbox_to_3d()
        self.save_nifti(self._stacked_data, os.path.join(output_dir, '4d_stacked_nifti.nii.gz'))
        self.save_nifti(self._collapsed_data, os.path.join(output_dir, '3d_stacked_nifti.nii.gz'))
        
    def gen_mask(self, output_dir, fname='mask.nii.gz'):
        """
        Generate a mask from the collapsed bounding box data and save it.
        
        params:
            output_path (str) : full path of directory to save the resultant nifti in. Must exist. 
        """
        self.generate_bounding_box()
        self.add_niftis_to_bounding_box()
        self.collapse_bbox_to_3d()
        mask = self.collapsed_bbox_to_mask()
        self.save_nifti(mask, os.path.join(output_dir, fname))