from inference import SwinDRNetPipeline
from PIL import Image
import cv2
import numpy as np

# test
model_path = "models/model.pth"
rgb = np.array(Image.open("testdata/000200-color.png").convert('RGB'))
depth = np.array(cv2.imread("testdata/000200-depth.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
true = np.array(cv2.imread("testdata/000200-depth_true.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
ppl = SwinDRNetPipeline(model_path)
result = ppl.inference(rgb, depth)

# show
rgb_scaled = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
depth_scaled = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
result_scaled = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
true_scaled = cv2.normalize(true, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('Scaled RGB Image', rgb_scaled.astype(np.uint8))
cv2.imshow('Scaled Depth Image', depth_scaled.astype(np.uint8))
cv2.imshow('Scaled Repaired Depth Image', result_scaled.astype(np.uint8))
cv2.imshow('Scaled True Depth Image', true_scaled.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()