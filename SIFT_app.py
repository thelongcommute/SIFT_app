#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np


class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)
        self.sift = cv2.xfeatures2d.SIFT_create()

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        # TODO run SIFT on the captured frame
        frame_homog = self.homography(frame)
        pixmap = self.convert_cv_to_pixmap(frame_homog)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

    def homography(self, frame):
        # load template query image
        img_query = cv2.imread(self.template_path)
        gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
        gray_train = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # find keypoints and descriptors for query and trian img
        kp_query, des_query = self.sift.detectAndCompute(gray_query, None)
        kp_train, des_train = self.sift.detectAndCompute(gray_train, None)
        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # matches is a list of lists. Each sublist contains the match
        # object for the query image and the train image
        if des_train is None:
            return frame
        matches = flann.knnMatch(des_query, des_train, k=2)
        # only keep good matches
        good_points = []
        ratio_thresh = 0.6
        for m, n in matches:
            # compare distances of descriptors. Ratio test.
            if m.distance < ratio_thresh * n.distance:
                good_points.append(m)
        # homography
        if len(good_points) > 12:
            # position of matching points in query image
            query_pts = np.float32([kp_query[m.queryIdx].pt
                                   for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_train[m.trainIdx].pt
                                   for m in good_points]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(query_pts, train_pts,
                                              cv2.RANSAC, 5.0)
            # Perspective transform
            h, w = gray_query.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]
                             ).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            # Draw lines on train image
            homography = cv2.polylines(frame, [np.int32(dst)],
                                       True, (255, 0, 0), 3)
            return homography
        else:
            return frame


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
