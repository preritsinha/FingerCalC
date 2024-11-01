import cv2
import mediapipe as mp
import math

# Video capture settings
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Hand definitions
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

# Actually the program calculates distance between two fingers, we must give which fingers
calculated_distances = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

def get_operation_type(fingers):
    if fingers == 1:
        return "Addition"
    elif fingers == 2:
        return "Subtraction"
    elif fingers == 3:
        return "Multiplication"
    elif fingers == 4:
        return "Division"
    else:
        return None

def perform_calculation(op_type, num1, num2):
    if op_type == "Addition":
        return num1 + num2
    elif op_type == "Subtraction":
        return num1 - num2
    elif op_type == "Multiplication":
        return num1 * num2
    elif op_type == "Division":
        if num2 != 0:
            return num1 / num2
        else:
            return "Error: Division by zero is not allowed."
    else:
        return None

num1 = None
num2 = None
op_type = None
result = None
step = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # It's optional, we used mirror effect
    img = cv2.flip(img, 1)

    # BGR to RGB Color conversion
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hands to count
    results = hands.process(img_rgb)

    # Finger counter
    counter = 0

    # When record every fingers, this condition will use
    if results.multi_hand_landmarks:
        # Motions array for record positions of all fingers
        motions = []

        for handLms in results.multi_hand_landmarks:
            # Draw 20 landmarks
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                # Convert ratios to real positions
                cx, cy = int(lm.x * w), int(lm.y * h)
                motions.append([id, cx, cy])

        for item in calculated_distances:
            ydist1  = (motions[item[0]][2]-motions[0][2])**2
            xdist1  = (motions[item[0]][1]-motions[0][1])**2
            ydist2 = (motions[item[1]][2]-motions[0][2])**2
            xdist2 = (motions[item[1]][1]-motions[0][1])**2
            dist1 = math.sqrt(xdist1+ydist1)
            dist2 = math.sqrt(xdist2+ydist2)
            # If down landmark of finger y position bigger than upper:
            # The finger increases counter
            isFingerOpen = dist2 > dist1
            counter += 1 if isFingerOpen else 0

    if step == 0:
        cv2.rectangle(img, (60, 5), (1000, 40), (5, 5, 5), cv2.FILLED)
        cv2.putText(img, "Show fingers for operation type and press Space", (70, 30), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            op_type = get_operation_type(counter)
            step = 1
    elif step == 1:
        cv2.rectangle(img, (60, 5), (1000, 40), (5, 5, 5), cv2.FILLED)
        cv2.putText(img, "Show fingers for first number and press Space", (70, 30), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            num1 = counter
            step = 2
    elif step == 2:
        cv2.rectangle(img, (60, 5), (1000, 40), (5, 5, 5), cv2.FILLED)
        cv2.putText(img, "Show fingers for second number and press Space", (70, 30), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            num2 = counter
            result = perform_calculation(op_type, num1, num2)
            step = 3
    elif step == 3:
        cv2.rectangle(img, (60, 5), (1000, 160), (5, 5, 5), cv2.FILLED)
        cv2.putText(img, f"Operation: {op_type}", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, f"Num1: {num1}", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, f"Num2: {num2}", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, f"Result: {result}", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, "Press 'r' to reset", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        if cv2.waitKey(1) & 0xFF == ord('r'):
            num1 = None
            num2 = None
            op_type = None
            result = None
            step = 0

    # Draw rectangle and put text for counting operation
    cv2.rectangle(img, (5, 5), (55, 40), (5, 5, 5), cv2.FILLED)
    cv2.putText(img, str(counter), (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    # Show all of these
    cv2.imshow("Capture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
