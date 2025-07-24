import numpy as np
import cv2


# 滑鼠讀取圖片座標
def on_mouse(event, x, y, flags, param):
    # EVENT_LBUTTONDOWN 左鍵點擊
    if event == cv2.EVENT_LBUTTONDOWN:
        src_point.append([x, y])
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)  # 點擊後變色


# 變換矩陣計算
def warp_perspective_matrix(src_point_input, target_point_input):
    assert src_point_input.shape[0] == target_point_input.shape[0] and src_point_input.shape[0] >= 4

    nums = src_point_input.shape[0]
    a = np.zeros((2 * nums, 8))  # A * war matrix = B
    b = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        a_i = src_point_input[i, :]
        b_i = target_point_input[i, :]

        a[2 * i, :] = [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0] * b_i[0], -a_i[1] * b_i[0]]
        b[2 * i] = b_i[0]
        a[2 * i + 1, :] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0] * b_i[1], -a_i[1] * b_i[1]]
        b[2 * i + 1] = b_i[1]

    a = np.mat(a)
    warp_matrix_out = a.I * b  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 後處理
    warp_matrix_out = np.array(warp_matrix_out).T[0]
    warp_matrix_out = np.insert(warp_matrix_out, warp_matrix_out.shape[0], values=1.0, axis=0)
    warp_matrix_out = warp_matrix_out.reshape((3, 3))
    return warp_matrix_out


# 矩陣還原
def my_warp_perspective(warp_img_input, warp_matrix_input):
    print('perspective running...')
    # for c in range(warp_img.shape):
    fix_matrix = warp_matrix_input
    a_11 = fix_matrix[0, 0]
    a_12 = fix_matrix[0, 1]
    a_13 = fix_matrix[0, 2]
    a_21 = fix_matrix[1, 0]
    a_22 = fix_matrix[1, 1]
    a_23 = fix_matrix[1, 2]
    a_31 = fix_matrix[2, 0]
    a_32 = fix_matrix[2, 1]
    a_33 = fix_matrix[2, 2]
    fix_img_out = np.zeros((warp_img_input.shape))

    max_y = warp_img_input.shape[0] - 1
    max_x = warp_img_input.shape[1] - 1
    for height_i in range(warp_img_input.shape[0]):  # height or y
        for width_i in range(warp_img_input.shape[1]):  # width or x
            y_fix = height_i
            x_fix = width_i

            # get y by
            y_denominator = (a_31 * y_fix - a_21) * a_12 - (a_31 * y_fix - a_21) * a_32 * x_fix - (
                    a_31 * x_fix - a_11) * a_22 + (a_31 * x_fix - a_11) * a_32 * y_fix

            y = ((a_23 - a_33 * y_fix - ((a_31 * y_fix - a_21) * a_13) / (a_31 * x_fix - a_11) + (
                    (a_31 * y_fix - a_21) * a_33 * x_fix) / (a_31 * x_fix - a_11)) * (
                         a_31 * x_fix - a_11)) / y_denominator

            x = (a_12 * y + a_13 - (a_32 * y * x_fix + a_33 * x_fix)) / (a_31 * x_fix - a_11)

            y = int(np.round(y))
            x = int(np.round(x))
            if y < 0:
                y = 0
            elif y > max_y:
                y = max_y

            if x < 0:
                x = 0
            elif x > max_x:
                x = max_x

            c_i = 0
            fix_img_out[height_i, width_i, c_i] = warp_img_input[y, x, c_i]
            c_i = 1
            fix_img_out[height_i, width_i, c_i] = warp_img_input[y, x, c_i]
            c_i = 2
            fix_img_out[height_i, width_i, c_i] = warp_img_input[y, x, c_i]

    print('perspective done!')
    return fix_img_out


# # # # # # # # # # #
if __name__ == '__main__':
    src_point = [[99, 100], [471, 6], [472, 342], [4, 343]]  # 儲存透視變換前的四點座標
    target_point = [[0, 0], [475, 0], [475, 347], [0, 347]]  # 透視變換後的四點座標(整張圖片的大小)
    # x_max = y_max = x_min = y_min = 0

    img = cv2.imread('Snipaste_2022-10-17_22-11-30.png')
    cv2.imshow('img_o', img)

    # 透過滑鼠選點事件
    # cv2.setMouseCallback('image', on_mouse)  # 由左上角開始,順時針點選四個點
    # while 1:
    #     cv2.imshow("image", img)
    #     k = cv2.waitKey(1)
    #     if k == 13:  # enter鍵停止
    #         break
    # print(src_point)

    src_point = np.array(src_point)
    target_point = np.array(target_point)

    # 計算透視變換矩陣
    warp_matrix = warp_perspective_matrix(src_point, target_point)
    print(warp_matrix)

    # 還原圖片
    fix_img = my_warp_perspective(img, warp_matrix)
    fix_img = (fix_img).astype(np.uint8)

    cv2.imshow('img_result', fix_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
