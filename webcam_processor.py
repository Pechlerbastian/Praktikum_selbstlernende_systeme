import cv2
import mediapipe as mp
from model.boxTrackingImagenet.detector1K.detector1K import Detector1K


class WebcamProcessor:

    '''
    This method initialises the class parameter needed for the bounding boxes, webcam feed, finger point detection
    and image processing
    @author: Marcel Achner
    '''
    def __init__(self):
        self.backup_img = None
        self.WEBCAM_WIDTH = 800
        self.WEBCAM_HEIGHT = 600
        self.take_screenshot = False
        self.current_x1 = None
        self.current_y1 = None
        self.current_x2 = None
        self.current_y2 = None
        self.cropping_offset = 8

        # variables for usage of mediapipe cause needed in every method
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        return

    '''
    This function detects the hand landmarks recognized from the input image and visualizes the visible index finger
    tip with a small circle in the image before returning it. Furthermore here the bounding box coordinates are 
    saved when a finger tip is pointing on or in a bounding box.
    @author: Marcel Achner
    '''
    def detect_fingerpointing_webcam(self, frame, frame_width, frame_height, boxes_list):
        with self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6,
                                 max_num_hands=1) as hands:

            # flip img needs to be done because it is mirrored on screen
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks is not None:
                normalized_landmark = results.multi_hand_landmarks[0].landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pixel_coordinates_landmark = self.mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x,
                                                                                              normalized_landmark.y,
                                                                                              frame_width,
                                                                                              frame_height)
                try:
                    cv2.circle(frame, pixel_coordinates_landmark, 2, (255, 0, 0), -1)
                except:
                    return frame

                # the bounding boxes list, (x1, y1) top-left starting point, (x2, y2) bottom-right ending point
                for box in boxes_list:
                    x1 = int(box[2])
                    y1 = int(box[1])
                    x2 = int(box[0])
                    y2 = int(box[3])

                    # note: bounding box is calculated and drawn in a coordinate system starting from top-right to
                    # bottom-left
                    landmark_x_tmp = int(pixel_coordinates_landmark[0])
                    landmark_y_tmp = int(pixel_coordinates_landmark[1])
                    # adapt the finger coordinates to be calculated in the same way as the bounding boxes
                    landmark_x = frame_width - landmark_x_tmp
                    landmark_y = landmark_y_tmp
                    # check if the finger is pointing on or in an existing bounding box
                    if x2 <= landmark_x <= x1 and y1 <= landmark_y <= y2:
                        self.take_screenshot = True
                        self.current_x1 = x2
                        self.current_y1 = y1
                        self.current_x2 = x1
                        self.current_y2 = y2

        return frame

    '''
    This function captures the webcam feed and combines object tracking using bounding boxes with fingerpointing
    of the index tip to specify the object that should be classified. Finally a screenshot from the cropped object
    is taken and returned
    @author: Marcel Achner
    '''
    def detect_objects(self):
        # path where the fingerpointing selected object image should be written on
        image_path = "data/detect_object.png"
        # create the video stream input of the webcam
        cam = cv2.VideoCapture(0)  # 0=front-cam, 1=back-cam
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WEBCAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WEBCAM_HEIGHT)
        frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while True:
            ret, image = cam.read()
            # save the input image from the webcam feed to use this as backup when one want to remove bounding boxes
            if image is None:
                continue
            self.backup_img = image.copy()
            cv2.flip(self.backup_img, 1)
            if not ret:
                continue

            # downsize factor of the img
            scale_percent = 100

            # update the image in the video sream and use bounding boxes to detect when finger pointing on it
            img, boxes_list = self.track_imnet(img=image, scale_percent=scale_percent)
            img = self.detect_fingerpointing_webcam(frame=img, frame_width=frame_width, frame_height=frame_height,
                                                    boxes_list=boxes_list)
            cv2.imshow("", img)
            # take a screenshot of the current image and write it to a file that can be read afterwards
            if self.take_screenshot:
                self.take_screenshot = False
                # use the previously saved image without bounding boxes for cropping and showing object
                cropped_img = self.crop_image()
                resized_img = self.padding_and_resize(cropped_img)
                cv2.flip(resized_img, 1)
                cv2.imwrite(image_path, resized_img)
                break  # close camera

            key = cv2.waitKey(30)
            # close camera with esc
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        return image_path

    '''
    This function uses the feature extraction model to get the shown objects and draws bounding boxes for each 
    object. At first downsize the given image to make predictions and bouding box drawings more efficient
    @author: Marcel Achner
    '''
    @staticmethod
    def track_imnet(img, scale_percent):
        detection_model_path = 'model/boxTrackingImagenet/models/object_localizer_float32.onnx'
        detection_threshold = 0.22

        detector = Detector1K(detection_model_path, detection_threshold)

        # downsize the image read from the webcam to ensure a faster processing of object tracking
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim_new = (width, height)
        # resize and downsize the image
        resized_img = cv2.resize(img, dim_new, interpolation=cv2.INTER_AREA)
        # detect objects on the smaller image
        detections = detector(resized_img)
        # draw detections on the original image
        detection_img, boxes_list = detector.draw_detections(img, scale_percent)

        return detection_img, boxes_list

    '''
    This function crops out the currently selected bounding box plus some more pixel to make sure to receive the 
    whole area of interest
    @author: Marcel Achner
    '''
    def crop_image(self):
        x_lower = int(self.current_x1 - self.cropping_offset)
        x_upper = int(self.current_x2 + self.cropping_offset)
        y_lower = int(self.current_y1 - self.cropping_offset)
        y_upper = int(self.current_y2 + self.cropping_offset)

        cropped_image = self.backup_img[y_lower:y_upper, x_lower:x_upper]

        return cropped_image

    '''
    This function uses the padding function for adding a padding to the smaller side of the image and then resizes 
    the image to the desired output size of 224x224
    @author: Marcel Achner
    '''
    def padding_and_resize(self, img):
        img = self.padding(img)
        cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        return img

    '''
    This function adds a padding to the image so that it will be of equal height and width
    @author: Marcel Achner
    '''
    @staticmethod
    def padding(img):
        img_height = img.shape[0]
        img_width = img.shape[1]

        # setting the desired img size to the size of the larger side of the img
        desired_size = max(img_width, img_height)

        delta_width = desired_size - img_width
        delta_height = desired_size - img_height
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        return cv2.copyMakeBorder(img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT)
