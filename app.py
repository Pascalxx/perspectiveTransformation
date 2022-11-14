import numpy as np
import cv2  # pip install -i https://pypi.douban.com/simple opencv-python==4.5.3.56


# 滑鼠讀取圖片座標
def OnMouse(event, x, y, flags, param):
    # EVENT_LBUTTONDOWN 左鍵點擊
    if event == cv2.EVENT_LBUTTONDOWN:
        src_point.append([x, y])
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)  # 點擊後變色


# 繪製直線
def draw_line(img, lines):
    for line_points in lines:
        cv2.line(img, (line_points[0][0], line_points[0][1]), (line_points[0][2], line_points[0][3]),
                 (0, 255, 0), 2, 1, 0)


# 計算四條直線的焦點作為頂點座標
def computer_intersect_point(lines):
    def get_line_k_b(line_point):
        """
          計算直線的斜率和截距
          :param line_point:直線的座標點
          :return:
        """

        x1, y1, x2, y2 = line_point[0]  # 獲取直系的兩點座標

        # 計算直線的斜率和截距
        k = (y1 - y2) / (x1 - x2)
        b = y2 - x2 * (y1 - y2) / (x1 - x2)
        return k, b

    line_intersect = []  # 用來存放直線的交點座標
    for i in range(len(lines)):
        k1, b1 = get_line_k_b(lines[i])
        for j in range(i + 1, len(lines)):
            k2, b2 = get_line_k_b(lines[j])

            # 計算交點座標
            x = (b2 - b1) / (k1 - k2)
            y = k1 * (b2 - b1) / (k1 - k2) + b1
            if x > 0 and y > 0:
                line_intersect.append((int(np.round(x)), int(np.round(y))))

    return line_intersect


def draw_point(img, points):
    for position in points:
        cv2.circle(img, position, 5, (0, 0, 255), -1)

    while 1:
        cv2.imshow("line_img", img)
        k = cv2.waitKey(1)
        if k == 13:  # enter鍵停止
            break


# 變換矩陣計算
def warp_perspective_matrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A * war matrix = B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]

        A[2 * i, :] = [A_i[0], A_i[1], 1,
                       0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]
                       ]

        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0,
                           A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]
                           ]

        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warp_matrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 後處理
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warp_matrix = warp_matrix.reshape((3, 3))
    return warp_matrix


# 矩陣還原
def my_warp_perspective(warp_img, warp_matrix):
    # for c in range(warp_img.shape):
    fix_matrix = warp_matrix
    a_11 = fix_matrix[0, 0]
    a_12 = fix_matrix[0, 1]
    a_13 = fix_matrix[0, 2]
    a_21 = fix_matrix[1, 0]
    a_22 = fix_matrix[1, 1]
    a_23 = fix_matrix[1, 2]
    a_31 = fix_matrix[2, 0]
    a_32 = fix_matrix[2, 1]
    a_33 = fix_matrix[2, 2]
    fix_img = np.zeros((warp_img.shape))

    max_y = warp_img.shape[0] - 1
    max_x = warp_img.shape[1] - 1
    for height_i in range(warp_img.shape[0]):  # height or y
        for width_i in range(warp_img.shape[1]):  # width or x
            Y_fix = height_i
            X_fix = width_i

            # get y by
            y_denominator = (a_31 * Y_fix - a_21) * a_12 - (a_31 * Y_fix - a_21) * a_32 * X_fix - \
                            (a_31 * X_fix - a_11) * a_22 + (a_31 * X_fix - a_11) * a_32 * Y_fix
            y = ((a_23 - a_33 * Y_fix - ((a_31 * Y_fix - a_21) * a_13) / (a_31 * X_fix - a_11)
                  + ((a_31 * Y_fix - a_21) * a_33 * X_fix) / (a_31 * X_fix - a_11))
                 * (a_31 * X_fix - a_11)) / y_denominator

            x = (a_12 * y + a_13 - (a_32 * y * X_fix + a_33 * X_fix)) / (a_31 * X_fix - a_11)

            # print("height_i:",height_i,",width_i:",width_i,",y:",y,",fix_x:",x)
            y = int(np.round(y))
            x = int(np.round(x))
            if y < 0:
                y = 0
            if y > max_y:
                y = max_y
            if x < 0:
                x = 0
            if x > max_x:
                x = max_x

            c_i = 0
            fix_img[height_i, width_i, c_i] = warp_img[y, x, c_i]
            c_i = 1
            fix_img[height_i, width_i, c_i] = warp_img[y, x, c_i]
            c_i = 2
            fix_img[height_i, width_i, c_i] = warp_img[y, x, c_i]
            print("height_i:", height_i, ",width_i:", width_i, ",y:", y, "fix_x:", x)

    return fix_img


# # # # # # # # # # #
if __name__ == '__main__':
    src_point = []  # 儲存透視變換前的四點座標
    target_point = [[0, 0], [475, 0], [475, 347], [0, 347]]  # 透視變換後的四點座標(整張圖片的大小)
    # x_max = y_max = x_min = y_min = 0

    img = cv2.imread('Snipaste_2022-10-17_22-11-30.png')
    cv2.namedWindow('image')

    # setMouseCallback 用來處理鼠標動作的函數
    # 當鼠標事件觸發時，OnMouse()回調函數會被執行
    cv2.setMouseCallback('image', OnMouse)  # 由左上角開始,順時針點選四個點

    while 1:
        cv2.imshow("image", img)
        k = cv2.waitKey(1)
        if k == 13:  # enter鍵停止
            break

    print(src_point)

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 將原圖轉為灰階圖
    # canny_img = cv2.Canny(gray_img, 100, 150, 3)  # Canny邊緣檢測
    # lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 70, minLineLength=180, maxLineGap=10)  # Hough直線檢測
    # draw_line(img, lines)  # 基於邊緣檢測的圖像來檢測直線
    #
    # line_intersect = computer_intersect_point(lines)  # 計算直線的交點座標
    # draw_point(img, line_intersect)  # 繪製交點座標的位置

    src_point = np.array(src_point)
    target_point = np.array(target_point)

    warp_matrix = warp_perspective_matrix(src_point, target_point)  # 計算透視變換矩陣
    print(warp_matrix)

    # for i in range(0, 475):
    #     for j in range(0, 347):
    #         img[j, i] = img[j, i].dot(warp_matrix)

    fix_img = my_warp_perspective(img, warp_matrix)  # 還原圖片
    fix_img = (fix_img).astype(np.uint8)
    # processed = cv2.warpPerspective(img, warp_matrix, (475, 347))     # 運用cv套件

    while (True):
        cv2.imshow('frame', fix_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    testAry = np.array([471, 10, 1])
    testResult = warp_matrix.dot(testAry)
    print(testResult)
