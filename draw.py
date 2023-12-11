import cv2


def draw_box(img, bbox, index_list, save_path):
    print(index_list)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (255, 255, 0)
    ]
    for idx in range(len(bbox)):
        i = bbox[index_list[idx]]
        start = (int(i[0]), int(i[1]))
        end = (int(i[2]), int(i[3]))
        img = cv2.rectangle(img, start, end, colors[idx], 2)
        img = cv2.putText(img, str(idx), (int(i[0]), int(i[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imwrite(save_path, img)