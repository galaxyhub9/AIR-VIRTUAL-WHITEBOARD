from django.shortcuts import render
from django.http import JsonResponse

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
# Create your views here.

def run_canvas(request):


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

    # The kernel to be used for dilation purpose
    kernel = np.ones((5, 5), np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]  # Added white for eraser
    colorIndex = 0

    # Here is code for Canvas setup
    paintWindow = np.zeros((720, 800,3)) + 255
    cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
    cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
    cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
    cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)
    cv2.rectangle(paintWindow, (620, 1), (750, 65), (0, 0, 0), 2)  # Eraser button
    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "ERASER", (720, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    h,w= 720,1100
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ret = True
    while ret:
        # Read each frame from the webcam
        ret, frame = cap.read()
        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
        cv2.rectangle(frame, (620, 1), (720, 65), (0, 0, 0), 2)  # Eraser button
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "ERASER", (640, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


        # Get hand landmark prediction
        result = hands.process(framergb)

        # Post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            # Handle color selection and drawing
            if thumb[1] - center[1] < 30:
                # Add new deques for drawing points
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    print("blue mode on ")
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                    print("green mode on ")
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                    print("red mode on ")
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
                    print("yellow mode on ")
                elif 605 <= center[0] <= 750:
                    colorIndex = 4  # Eraser
                    print("erase mode on ")
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(fore_finger)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(fore_finger)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(fore_finger)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(fore_finger)
                elif colorIndex == 4:
                    print("color index is 4-----------")
                    cv2.circle(frame, fore_finger,20,colors[4], -1)
                    # cv2.circle(paintWindow, fore_finger,50,colors[4], -1)
                    eraser_radius = 50  # The radius of the eraser
                    for points in [bpoints, gpoints, rpoints, ypoints]:
                        for i in range(len(points)):
                            # Split the deque into multiple deques at the points that are under the eraser
                            new_deques = []
                            current_deque = deque(maxlen=512)
                            for point in points[i]:
                                print(f"point: {point}, fore_finger: {fore_finger}")

                                if np.linalg.norm(np.array(point) - np.array(fore_finger)) <= eraser_radius:
                                    if current_deque:
                                        new_deques.append(current_deque)
                                        current_deque = deque(maxlen=512)
                                else:
                                    current_deque.append(point)
                            if current_deque:
                                new_deques.append(current_deque)
                            points[i] = deque(point for deque in new_deques for point in deque)
        
        points = [bpoints, gpoints, rpoints, ypoints]                                
        paintWindow.fill(255)  # Clear the paint window  
        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue                
                    else:
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Display the frame and paint window
        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        # Check for key press to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    response_data = {'message': 'Air Canvas executed successfully!'}
    return JsonResponse(response_data)

def canvas_view(request):
    return render(request,'home.html' )