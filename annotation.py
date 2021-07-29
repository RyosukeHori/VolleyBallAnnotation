import os
import cv2
import numpy as np
import glob
import json
import argparse
import time
from pynput import keyboard
from matplotlib import pyplot as plt
from scipy import interpolate
import json

parser = argparse.ArgumentParser(description='Volley Ball Annotation')
parser.add_argument('video_path', help='Video path')
parser.add_argument('out_path', help='Jason output path')

args = parser.parse_args()

video_path = args.video_path
out_path = args.out_path


def createimage(w, h):
    size = (h, w, 3)
    img = np.ones(size, np.uint8) * 255
    cv2.line(img, (124, 62), (1924, 62), (0, 0, 0))
    cv2.line(img, (124, 962), (1924, 962), (0, 0, 0))
    cv2.line(img, (124, 62), (124, 962), (0, 0, 0))
    cv2.line(img, (724, 62), (724, 962), (0, 0, 0))
    cv2.line(img, (1024, 62), (1024, 962), (0, 0, 0))
    cv2.line(img, (1324, 62), (1324, 962), (0, 0, 0))
    cv2.line(img, (1924, 62), (1924, 962), (0, 0, 0))
    cv2.circle(img, (1024, 512), 6, (0, 0, 0), -1)
    return img


def annotate():
    #print(player_set)
    print('Please select player id:', end=' ')
    player_id = input()
    #player_set.remove(player_id)
    player_list.append(player_id)
    player_pos[player_id] = list()
    reset_key = False

    for fr_idx in range(frame_num // 10):
        print('fr: ', fr_idx * 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx * 10)
        ret, frame = cap.read()
        if not ret:
            raise IOError('Error: image capture failed!')

        window_name = 'img_' + str(fr_idx)
        position = [0, 0]
        def onMouse(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                #print(x, y)
                position[0] = x
                position[1] = y

        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, onMouse)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        player_pos[player_id].append(position)
        #print(player_pos)



if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError('Error: video import failed!')
    print("Video loaded:", video_path)
    v_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    v_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    player_pos = dict()
    player_list = []

    #ret, frame = cap.read()
    #cv2.imshow('players', frame)
    #print('Players ID:', end='')
    #player_set = set(input().split())

    # set players' position to player_pos
    for idx in range(3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        annotate()

    #print(player_pos)

    print('-' * 30)
    print('Calculate Homography Matrix.')
    # get Homography Matrix and transform positions
    p_original = []
    corner_pos = []
    window_name = 'GetCornerPos'

    def getCornerPos(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            corner_pos= []
            corner_pos.append(x)
            corner_pos.append(y)
            p_original.append(corner_pos)

    while(1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, getCornerPos)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        if len(p_original) == 4:
            break
        else:
            p_original = []
            print('Please select 4 corners.')

    '''
    for corner in p_original:
        cv2.circle(frame, tuple(corner), 3, (0, 255, 0), -1)
    cv2.imshow('warpImage', frame)
    cv2.waitKey(0)
    cv2.destroyWindow()
    '''

    p_original = np.float32(p_original)
    p_trans = np.float32([[124, 62], [1924, 62], [1924, 962], [124, 962]])

    M = cv2.getPerspectiveTransform(p_original, p_trans)

    trans_frame = createimage(2048, 1024)
    for p_id in player_list:
        im_pos = np.insert(np.array(player_pos[p_id]), 2, 1, axis=1)
        trans_pos = np.dot(M, im_pos.T)
        trans_pos = trans_pos / trans_pos[2]
        trans_pos = trans_pos.T[:, :2]

        # interpolation
        x = trans_pos[:, 0]
        y = trans_pos[:, 1]
        x_latent = np.linspace(0, 1, frame_num, endpoint=True)
        fitted_curve, _ = interpolate.splprep([x, y], k=3, s=0)
        trans_pos = interpolate.splev(x_latent, fitted_curve)
        '''
        plt.scatter(x, y, label="observed")
        plt.plot(trans_pos[0], trans_pos[1], c="red", label="fitted")
        plt.grid()
        plt.legend()
        plt.show()
        '''
        player_pos[p_id] = np.vstack((trans_pos[0], trans_pos[1])).T


        for pos in player_pos[p_id]:
            cv2.circle(trans_frame, (int(pos[0]), int(pos[1])), 3, (0, 255, 0), -1)

    window_name = 'warpImage'
    cv2.imshow(window_name, trans_frame)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    # seve position data to Json file
    print('Saving data to json.')
    for frame_idx in range(frame_num):
        print('frame: ', frame_idx)
        out_json = dict()
        out_json['img_idx'] = frame_idx

        pos_list = []
        for player_idx in player_list:
            player_dict = dict()
            player_dict['idx'] = int(player_idx)
            player_dict['pos'] = player_pos[player_idx][frame_idx].tolist()
            pos_list.append(player_dict)

        out_json['playerPos'] = pos_list
        if not os.path.exists(out_path):
            print("made dir:", out_path)
            os.makedirs(out_path)
        with open(out_path + str(frame_idx).zfill(6) + '.json', 'w') as fp:
            json.dump(out_json, fp, indent=4, ensure_ascii=False)

    print('Saving done!')