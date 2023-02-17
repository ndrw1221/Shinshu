# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
# pipeline.start(config)
# Depth camera inter para
profile = pipeline.start(config)
depth_intrin = (
    profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
)
try:

    paused = False
    x_divide = 0.2
    y_divide = 0.2

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Object detection code, to find the target point (pixel)
        # TODO
        # pixel = []
        # Experimental stage, using image center as target point.

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        # images.shape == depth_colormap.shape
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(
                color_image,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA,
            )
            # TODO
            # pixel_depth = depth_img.get_distance(pixel*depth.shape/color_img.shape)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            # TODO
            # pixel_depth = depth_img.get_distance(pixel)
            images = np.hstack((color_image, depth_colormap))
        # Convert point from color coordinate (pixel) into actuall 3D depth coordinate
        # TODO
        # Center point
        x, y = int(depth_colormap_dim[1] / 2), int(depth_colormap_dim[0] / 2)
        pixel_depth = depth_frame.get_distance(x, y)
        cv2.circle(images, (x, y), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.circle(
            images,
            (x + depth_colormap_dim[1], y),
            radius=5,
            color=(0, 255, 255),
            thickness=1,
        )
        cv2.putText(
            images,
            f"Depth: {round(pixel_depth, 2)}m.",
            (x, y + 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            images,
            f"Depth: {round(pixel_depth, 2)}m.",
            (x + depth_colormap_dim[1], y + 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # Other point
        x1, y1 = int(depth_colormap_dim[1] * x_divide), int(
            depth_colormap_dim[0] * y_divide
        )
        pixel_depth = depth_frame.get_distance(x1, y1)
        cv2.circle(images, (x1, y1), radius=5, color=(255, 0, 255), thickness=-1)
        cv2.circle(
            images,
            (x1 + depth_colormap_dim[1], y1),
            radius=5,
            color=(255, 0, 255),
            thickness=1,
        )
        depth_point_float = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [x1, y1], pixel_depth
        )
        depth_point = [round(num, 2) for num in depth_point_float]
        cv2.putText(
            images,
            f"Coordinate: {depth_point}m.",
            (x1, y1 + 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            images,
            f"Coordinate: {depth_point}m.",
            (x1 + depth_colormap_dim[1], y1 + 30),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        # print(depth_point)

        # Show images
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        if not paused:
            cv2.imshow("RealSense", images)
        key = cv2.waitKey(1)

        if key == ord("w") and y_divide > 0.1:
            y_divide -= 0.1

        if key == ord("a") and x_divide > 0.1:
            x_divide -= 0.1

        if key == ord("s") and y_divide < 0.9:
            y_divide += 0.1

        if key == ord("d") and x_divide < 0.9:
            x_divide += 0.1

        if key == ord("p"):
            paused = not paused

        if key == ord("o"):
            cv2.imwrite("./out.png", images)

        if key == 27 or key == ord("q"):  # Esc
            break

finally:

    # Stop streaming
    pipeline.stop()


# Added pause frame feature
