from torch.utils.data import DataLoader
import cv2 as cv
import numpy as np
import pathlib
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights



class openCVPipeline:
    def __init__(self, output_dir, dataset):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = dataset

    def get_ROI(self, img_closed):
      # Find contours of images
      inverted = cv.bitwise_not(img_closed)
      contours, hierarchy = cv.findContours(inverted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

      # Initialize variables for the outermost bounding box
      min_x, min_y = float('inf'), float('inf')
      max_x, max_y = float('-inf'), float('-inf')

      # for contour in contours:
      # Define bounding rectangle for each contour
      x, y, w, h = cv.boundingRect(contours[0])

      # Update the minimum and maximum coordinates to get ROI from all the bounding rectangles
      min_x = min(min_x, x)
      min_y = min(min_y, y)
      max_x = max(max_x, x + w)
      max_y = max(max_y, y + h)

      roi = (min_x, min_y, max_x, max_y)

      return contours, roi

    def process_image(self, image):
      # Convert image to HSV color space
      hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

      # Remove noise with the non-linear median filter
      img_blur = cv.medianBlur(src=hsv, ksize=7)

      # Convert image to grayscale
      gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)

      # Binary image for opening
      binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

      # Opening and closing (dilation + erosion > erosion + dilation) to eliminate protrusions
      dilation_kernel = np.ones((7, 7), np.uint8)
      erosion_kernel = np.ones((3, 3), np.uint8)
      img_opened = cv.dilate(cv.erode(binary, erosion_kernel, iterations=1), dilation_kernel, iterations=1)
      img_closed = cv.erode(cv.dilate(img_opened, dilation_kernel, iterations=1), erosion_kernel, iterations=9)

      return img_closed

    def saveROI(self, image, index):
      cv.imwrite(self.output_dir / f"roi_{index}.png", image)

    def display_image(self, image):
      cv.imshow("Image", image)
      cv.waitKey(0)
      cv.destroyAllWindows()

    def run(self):
      dataloader = DataLoader(self.dataset, batch_size=1,
                              shuffle=False, num_workers=0)

      for i, sample in enumerate(dataloader):
          print(i, sample.size())

          img = sample[0].cpu().numpy().astype("uint8")
          
          img_opened = self.process_image(img)

          contours, roi = self.get_ROI(img_opened)
          cv.drawContours(img, contours, -1, (0, 255, 0), 2)
          cv.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 2)

          # self.display_image(img)
          cropped_img = img[roi[1]:roi[3], roi[0]:roi[2]]

          self.saveROI(img, i)
          self.saveROI(cropped_img, f"{i}_cropped")


class FasterRCNNPipeline:
    def __init__(self, output_dir, dataset, threshold=0.6,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = dataset
        self.threshold = threshold
        self.device = device

        # Load pretrained Faster R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)

    def preprocess(self, image):
        """Convert BGR numpy image to tensor on device"""
        tensor = torch.from_numpy(image).to(self.device)
        tensor = F.convert_image_dtype(tensor)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    def predict(self, tensor):
        """Run Faster R-CNN and return filtered boxes, scores, labels"""
        with torch.no_grad():
            outputs = self.model(tensor)

        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        return boxes[0], scores[0], labels[0]

    def draw_and_crop(self, image, boxes, scores, index):
        """Draw boxes and save both full image + cropped ROIs"""
        img_out = image.copy()
        for j, (box, score) in enumerate(zip([boxes], [scores])):
            x1, y1, x2, y2 = box.astype(int)
            cv.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img_out, f"{score:.2f}", (x1, y1 - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save cropped ROI
            roi = image[y1:y2, x1:x2]
            cv.imwrite(str(self.output_dir / f"roi_{index}_{j}.png"), roi)

        # Save annotated full image
        cv.imwrite(str(self.output_dir / f"annotated_{index}.png"), img_out)

    def run(self):
        dataloader = DataLoader(self.dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        for i, sample in enumerate(dataloader):
            print(i, sample.size())

            img = sample[0].cpu().numpy().astype("uint8")  # [H,W,3] BGR
            tensor = self.preprocess(img)
            boxes, scores, labels = self.predict(tensor)

            if len(boxes) > 0:
                self.draw_and_crop(img, boxes, scores, i)