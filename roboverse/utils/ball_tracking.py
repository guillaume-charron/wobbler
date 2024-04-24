from roboverse.utils.camera import Camera
import cv2
import imutils
import numpy as np
cam_width, cam_height = 1920, 1080
cam_src = 1
cam_fps = 30

def analyze_frame(frame, draw=False):
    plate_center_pos = None
    ball_center_pos = None
    # green color
    lower = (65, 61, 58)
    upper = (179, 255, 255)

    imutils.resize(frame, width=600)
    new_frame = frame.copy()
    gaussian = cv2.GaussianBlur(frame, (71, 71), 0)
    frame_hsv = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(frame_hsv, lower, upper)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_center_pos = None
    rect_bbox = None
    biggest_rec = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        threshold = 20
        if w > threshold and h > threshold:
            area = w * h
            if area > biggest_rec:
                biggest_rec = area
                rect_bbox = (w, h)

                if draw:
                    new_frame = cv2.drawContours(new_frame, [cnt], -1, (0,255,255), 3)
                # Find center of contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                if draw:
                    cv2.circle(new_frame, (cX, cY), 7, (255, 0, 0), -1)
                rect_center_pos = (cX, cY)

                # Find circle in crop image
                crop = frame[y:y+h, x:x+w]
                gaussian = cv2.GaussianBlur(crop, (81, 81), 0)
                gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
                _, thresh= cv2.threshold(gray, 205, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                biggest_ball = None
                biggest_radius = 0
                for c in contours:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        #####
                        (x_circle,y_circle),radius = cv2.minEnclosingCircle(c)
                        center = (int(x_circle),int(y_circle))
                        
                        radius = int(radius)
                        cX = int(M["m10"] / M["m00"])
                
                        cY = int(M["m01"] / M["m00"])
                        if radius > biggest_radius and radius > 5:
                            biggest_radius = radius
                            biggest_ball = (cX, cY, radius)

                if biggest_ball:
                    cX, cY, radius = biggest_ball
                    if draw:
                        cv2.circle(new_frame, (int(cX + x), int(cY + y)), 5, (0,255,0), -1)
                    plate_center_pos = rect_center_pos
                    ball_center_pos = (int(cX + x), int(cY + y))
    if ball_center_pos is not None:
        relative_pos = np.array(plate_center_pos) - np.array(ball_center_pos)
        relative_pos = relative_pos / np.maximum(rect_bbox[0], rect_bbox[1]) * 0.2
        return relative_pos, new_frame
    return None, new_frame


if __name__ == "__main__":
    cam = Camera(0, cam_width, cam_height, cam_fps)
    while True:
        frame = cam.get_frame()
        if frame is not None:
            relative_pos, new_frame = analyze_frame(frame.copy(), draw=True)
            print(relative_pos)
            if relative_pos is not None:
                print(np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2))
            new_frame = cv2.resize(new_frame, (600, 400))
            cv2.imshow('frame', new_frame)
            #cv2.imwrite('frame_green.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop_cam()