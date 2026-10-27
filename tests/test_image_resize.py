import unittest
import numpy as np
from bimanual_vla.data.image_resize import resize_stretch, StretchResizeImages, PreparedImages
from bimanual_vla.collection.camera import CameraCapture

class ImageResizeTest(unittest.TestCase):
    def test_training_only_validates_prepared_images(self):
        frame = np.full((224,224,3), 100, dtype=np.uint8)
        data = {'image': {'cam_high': frame}}
        self.assertIs(PreparedImages(224,224)(data)['image']['cam_high'], frame)
        with self.assertRaisesRegex(ValueError, 'prepared uint8 RGB'):
            PreparedImages(224,224)({'image': {'cam_high': np.zeros((240,424,3),np.uint8)}})

    def test_new_capture_defaults_to_ready_no_padding_images(self):
        class Cap:
            def read(self):
                return True, np.full((240,424,3), 100, dtype=np.uint8)
        camera = CameraCapture(cam_ids={'cam_high':'fake'})
        camera._caps = {'cam_high': Cap()}
        images, _ = camera._read_direct()
        self.assertEqual(images['cam_high'].shape, (3,224,224))
        self.assertEqual(int(images['cam_high'].min()), 100)

    def test_native_policy_resize_skips_recording_resolution(self):
        rgb = np.random.default_rng(42).integers(0, 256, (240, 424, 3), dtype=np.uint8)
        class Cap:
            def read(self):
                return True, rgb[..., ::-1].copy()
        # Production records at 256; policy input must resize native RGB once,
        # including when the legacy pre-resize optimization is disabled.
        for preresize in (None, (224, 224), (256, 256)):
            camera = CameraCapture(cam_ids={'cam_high':'fake'}, image_hw=(256,256),
                                   policy_image_hw=preresize)
            camera._caps = {'cam_high': Cap()}
            camera.resize_mode = 'stretch'
            recorded, _ = camera._read_direct()
            policy = camera._prepare_policy_images(recorded)['cam_high'].transpose(1,2,0)
            policy_hw = preresize or (224, 224)
            expected = StretchResizeImages(*policy_hw)({'image':{'cam_high':rgb}})['image']['cam_high']
            np.testing.assert_array_equal(policy, expected)
            self.assertEqual(recorded['cam_high'].shape, (3,256,256))
            twice = resize_stretch(recorded['cam_high'].transpose(1,2,0),224,224)
            if policy_hw == (224, 224):
                self.assertFalse(np.array_equal(policy, twice))

    def test_training_and_camera_use_same_no_padding_transform(self):
        image=np.full((240,424,3),120,dtype=np.uint8)
        image[:,:60]=[220,100,80];image[:,-60:]=[80,100,220]
        expected=StretchResizeImages(224,224)({'image':{'cam_high':image}})['image']['cam_high']
        camera=CameraCapture(policy_image_hw=(224,224));camera.resize_mode='stretch'
        actual=camera._prepare_policy_images({'cam_high':image.transpose(2,0,1)})['cam_high'].transpose(1,2,0)
        np.testing.assert_array_equal(expected,actual)
        self.assertGreater(int(actual.min()),0)
        self.assertGreater(int(actual[100,0,0]),200)
        self.assertGreater(int(actual[100,-1,2]),200)
        np.testing.assert_array_equal(resize_stretch(actual,224,224),actual)

    def test_raw_camera_does_not_add_letterbox_in_stretch_mode(self):
        class Cap:
            def read(self):return True,np.full((240,424,3),120,dtype=np.uint8)
        camera=CameraCapture(cam_ids={'cam_high':'fake'},image_hw=(224,224))
        camera._caps={'cam_high':Cap()};camera.resize_mode='stretch'
        images,_=camera._read_direct()
        self.assertEqual(images['cam_high'].shape,(3,224,224))
        self.assertEqual(int(images['cam_high'].min()),120)
