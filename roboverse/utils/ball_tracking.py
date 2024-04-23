from roboverse.utils.camera import Camera
import cv2
import imutils
cam_width, cam_height = 1920, 1080
cam_src = 0
cam_fps = 30

def analyze_frame(frame, draw=False):
    plate_center_pos = None
    ball_center_pos = None
    # green color
    lower = (59, 59, 165)
    upper = (95, 107, 255)

    imutils.resize(frame, width=600)
    new_frame = frame.copy()
    gaussian = cv2.GaussianBlur(frame, (71, 71), 0)
    frame_hsv = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(frame_hsv, lower, upper)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_center_pos = None
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            threshold = 10
            if w > threshold and h > threshold:
                if ratio >= 0.9 and ratio <= 1.1 and draw:
                    new_frame = cv2.drawContours(new_frame, [cnt], -1, (0,255,255), 3)
                    cv2.putText(new_frame, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                elif draw:
                    new_frame = cv2.drawContours(new_frame, [cnt], -1, (0,255,0), 3)
                    cv2.putText(new_frame, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)    

                # Find center of contour
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
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
                    break
        return plate_center_pos, ball_center_pos, new_frame


if __name__ == "__main__":
    cam = Camera(cam_src, cam_width, cam_height, cam_fps)
    while True:
        frame = cam.get_frame()
        if frame is not None:
            plate_center_pos, ball_center_pos, new_frame = analyze_frame(frame, draw=True)
            cv2.imshow('frame', new_frame)
            cv2.imwrite('frame_green.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop_cam()