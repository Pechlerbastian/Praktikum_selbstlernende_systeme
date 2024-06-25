import cv2
import numpy as np
import onnxruntime

'''
This class provides the functionality of the feature extraction model to detect and track objects and draw bounding
boxes.
@author: Marcel Achner
'''


class Detector1K:
    '''
    This method initializes the needed class parameters, the detection model, and the classification model
    @author: Marcel Achner
    '''
    def __init__(self, detector_model_path, detector_threshold=0.2):

        self.input_width = None
        self.input_height = None
        self.input_shape = None
        self.input_name = None
        self.session = None
        self.output_names = None
        self.img_channels = None
        self.img_width = None
        self.img_height = None
        self.label_detections = None
        self.threshold = detector_threshold

        # Initialize detection model
        self.initialize_model(detector_model_path)

        # Set the crop offset
        self.crop_offset = 0

    '''
    This method defines the call to detect objects in the given image
    @author: Marcel Achner
    '''
    def __call__(self, image):

        return self.detect_objects(image)

    '''
    This function is responsible for the initialisation of the model to detect objects
    @author: Marcel Achner
    '''
    def initialize_model(self, model_path):

        self.session = onnxruntime.InferenceSession(model_path)

        # Get model info
        self.get_model_input_details()
        self.get_model_output_details()

    '''
    This function gets invoked by the __call__ method and it prepares the image as input and performs inference on it.
    The result is used to find detections in the image which get pre-classified then. In the end the label of the 
    detections are returned.
    @author: Marcel Achner
    '''
    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        detections = self.process_output(outputs)

        # Set the label detections
        self.label_detections = detections

        return self.label_detections

    '''
    This function performs different modifications on the input image: resizing, changing BGR to RGB color
    and transposing
    @author: Marcel Achner
    '''
    def prepare_input(self, img):

        self.img_height, self.img_width, self.img_channels = img.shape

        # Transform the image for inference
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    '''
    This method performs the inference on the image using the input name and the output names as parameter
    @author: Marcel Achner
    '''
    def inference(self, input_tensor):

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs

    '''
    This method processes the output and calculates the bounding box coordinates if the confidence >= threshold.
    Then it constructs a JSON-Object as return object with the bounding box coordinates and some more parameters.
    @author: Marcel Achner
    '''
    def process_output(self, outputs):

        # Get all output details
        boxes = np.squeeze(outputs[0])
        scores = np.squeeze(outputs[2])
        num_objects = int(outputs[3][0])

        results = []
        for i in range(num_objects):
            if scores[i] >= self.threshold:
                y1 = (self.img_height * boxes[i][0]).astype(int)
                y2 = (self.img_height * boxes[i][2]).astype(int)
                x1 = (self.img_width * boxes[i][1]).astype(int)
                x2 = (self.img_width * boxes[i][3]).astype(int)

                result = {
                    'bounding_box': np.array([x1, y1, x2, y2], dtype=int),
                    'class_id': 0,
                    'label': "",
                    'detection_score': scores[i],
                    'classification_score': 0,
                }
                results.append(result)
        return results

    '''
    This function sets class parameter with information from the input of the model
    @author: Marcel Achner
    '''
    def get_model_input_details(self):

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    '''
    This function sets class parameter with information from the output of the model
    @author: Marcel Achner
    '''
    def get_model_output_details(self):

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    '''
    This method uses the detected objects in an image to draw the corresponding bounding boxes around them.
    For a better performance only the objects of interest are detected when they are not detected on the 
    very edge of the window. This won´t recognize human bodies or faces in the given image.
    @author: Marcel Achner
    '''
    def draw_detections(self, image, scale_percent):
        boxes_list = []

        for idx, detection in enumerate(self.label_detections):

            # calculate boxes for original image size and use original image size for next steps
            # check if the object is from the surrounding and do not draw boxes then
            box = detection['bounding_box']
            if int(box[3]) + (30 * (scale_percent / 100)) >= image.shape[0] / (100 / scale_percent):
                continue
            # rescale the boxes back to the original size
            box = np.multiply(box, (100 / scale_percent))
            boxes_list.append(box)

            color = (255, 0, 0)
            text_thickness = int(min([self.img_height, self.img_width]) * 0.004)
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, text_thickness)

        return image, boxes_list
