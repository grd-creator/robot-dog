import cv2
import numpy as np

colors = np.random.uniform(0, 255, size=(9, 3))
def draw_bounding_box(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{classes[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_frame(frame, onnx_model, classes):
    model = cv2.dnn.readNetFromONNX(onnx_model)
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640  # 缩放比例

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    outputs = model.forward()
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    class_label = None
    box_max=[]
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        draw_bounding_box(frame, classes, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        if class_ids is not None:
            max_area = 0
            max_index = -1
            box_max = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                area = box[2] * box[3]
                if area > max_area:
                    max_area = area
                    max_index = index
                    class_label = class_ids[max_index]
                    box_max = box

    return frame, class_label, box_max


def get_status(box, box1, image):
    dashboard_status = None
    if len(box)>0 and len(box1)>0:
        x = box[0] + box[2] / 2
        y = box[1] + box[3] / 2
        # 转换坐标
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        box1[2] = box1[0] + box1[2] / 2
        box1[3] = box1[1] + box1[3] / 2

        def test(lst, num):
            # 使用 int() 将 np.ceil 的结果转换为整数
            return list(map(lambda x: lst[x * num: x * num + num], list(range(0, int(np.ceil(len(lst) / num))))))

        box = test(box, 2)

        # 取完整圆的60%部分
        center = np.mean(box, axis=0)
        scale_bbox = center + (box - center) * 0.5
        scale_bbox = scale_bbox.astype(np.int32)
        # 图片转换为灰度图
        image = image[scale_bbox[0][1]:scale_bbox[1][1], scale_bbox[0][0]:scale_bbox[1][0], :]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化，将黑色区域设为255，其他区域设为0
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到面积最大的轮廓
        max_area = 0
        max_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                max_contour = c
        # 如果找到了最大轮廓，绘制它并计算它的角度
        if max_contour is not None:
            # cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
            rect = cv2.minAreaRect(max_contour)
            wucha = 5
            wucha_f = -5
            # 得到指针轮廓的最佳拟合直线，并获取其斜率
            vx, vy, x0, y0 = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy / vx
            if x != box1[2]:
                slope1 = (y - box1[3]) / (x - box1[2])
                if np.isinf(slope):
                    angle = 90.0
                else:
                    angle = np.arctan(slope) * 180 / np.pi
                # 求指针偏转角度
                angle = 270 + angle
                cv2.imshow('image', image)
                if (rect[0][0] - image.shape[1] // 2)<= wucha and (rect[0][0] - image.shape[1] // 2)>= wucha_f:
                    angle = 180
                    if rect[0][1] > image.shape[1] // 2:
                        angle = 0
                elif rect[0][0] < image.shape[1] // 2:
                    angle -= 180
                # 求零点偏转角度
                # 处理垂直线
                if np.isinf(slope1):
                    angle1 = 90.0
                else:
                    angle1 = np.arctan(slope1) * 180 / np.pi
                angle1 = 270 + angle1
                if  box1[2]-x <= wucha and box1[2]-x >= wucha_f:
                    angle1 = 180
                    if box1[3] > y:
                        angle1 = 0
                elif box1[2] < x:
                    angle1 -= 180
                # 计算零点偏差，从而修正指针角度
                angel_fina = (angle - angle1) % 360
                print(angel_fina)
            if 0 <= angel_fina < 129:
                dashboard_status = "偏低"
            elif 129 <= angel_fina < 238:
                dashboard_status = "正常"
            elif 238 <= angel_fina < 360:
                dashboard_status = "偏高"
        else:
            dashboard_status = None
    return dashboard_status

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    model_clock = 'best_clock_far.onnx'
    classes_clock = {0:'clock'}
    model_0 = 'best_0.onnx'
    classes_0 = {0:'ssi'}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame_clock, class_label_clock, box_clock = process_frame(frame, model_clock,classes_clock)
        cv2.imshow('Camera', processed_frame_clock)
        processed_frame_0, class_label_0, box_0 = process_frame(frame, model_0,classes_0)
        print(class_label_0)
        cv2.imshow('Camera', processed_frame_0)
        clock_status = get_status(box_clock, box_0, frame)
        print(clock_status)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()