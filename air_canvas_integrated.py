# aircanvas.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

#The kernel to be used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')

# Add the getTool function from virtualpaint.py
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x<100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return"draw"
    elif x<200 + ml:
        return "circle"
    else:
        return "erase"

# Add the index_raised function from virtualpaint.py
def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

cap = cv2.VideoCapture(0)
ret = True
# Modify the main loop
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            x = int(handslms.landmark[8].x*640)
            y = int(handslms.landmark[8].y*480)
            # Add the tool selection code from virtualpaint.py
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()
                cv2.circle(frame, (x, y), rad, (0,255,255), 2)
                rad -= 1
                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("your current tool set to : ", curr_tool)
                    time_init = True
                    rad = 40

            # Add the tool usage code from virtualpaint.py
            if curr_tool == "draw":
                xi, yi = int(handslms.landmark[12].x*640), int(handslms.landmark[12].y*480)
                y9  = int(handslms.landmark[9].y*480)
                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y
            elif curr_tool == "line":
                xi, yi = int(handslms.landmark[12].x*640), int(handslms.landmark[12].y*480)
                y9  = int(handslms.landmark[9].y*480)
                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frame, (xii, yii), (x, y), (50,152,255), thick)
            elif curr_tool == "rectangle":
                xi, yi = int(handslms.landmark[12].x*640), int(handslms.landmark[12].y*480)
                y9  = int(handslms.landmark[9].y*480)
                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frame, (xii, yii), (x, y), (0,255,255), thick)
            elif curr_tool == "circle":
                xi, yi = int(handslms.landmark[12].x*640), int(handslms.landmark[12].y*480)
                y9  = int(handslms.landmark[9].y*480)
                if index_raised(yi, y9):
                    if not(var_inits):
                        xii, yii = x, y
                        var_inits = True
                    cv2.circle(frame, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)
            elif curr_tool == "erase":
                xi, yi = int(handslms.landmark[12].x*640), int(handslms.landmark[12].y*480)
                y9  = int(handslms.landmark[9].y*480)
                if index_raised(yi, y9):
                    cv2.circle(frame, (x, y), 30, (0,0,0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Blue
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # Red
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 # Yellow
        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    # Append the next deques when nothing is detected to avois messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    frame[:, :, 1] = result[:, :, 1]
    frame[:, :, 2] = result[:, :, 2]
    frame[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frame[:max_y, ml:max_x], 0.3, 0)
    cv2.putText(frame, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	

    # ... rest of the aircanvas.py main loop code ...
    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)
    

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
