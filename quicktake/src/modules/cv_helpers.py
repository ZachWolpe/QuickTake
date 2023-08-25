'''
========================================================
Helper functions for OpenCV.

: zach wolpe, 19 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''
import cv2
# opencv helpers ------------------------------------------------------------------------->>
class CVHelpers:
# add block to image
    @staticmethod
    def add_block_to_image(image, text, x0,y0,x1,y1, colour=(255,0,0), thickness=3):
        """updates image inplace"""
        text_padding = 20
        try:
            cv2.rectangle(img=image, pt1=(x0,y0),   pt2=(x1,y1),    color    = colour, thickness=thickness)
            cv2.putText(img=image,   text=text,     org=(x0,y1-text_padding),    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colour, thickness=thickness)
        except:
            print('Error adding block to image!')

    @staticmethod
    def generate_yolo_label(flags:list[set]):
        # flag structure:
        # flags = [('label', 'value', 'show_label', 'round')]
        # example: [('person', 'person', False, False),
        #           ('confidence', 0.99, True, True)
        # ]
        label = "("
        for flag in flags:
            _label = flag[0] + ": "     if flag[2] else ""
            _value = round(flag[1], 2)  if flag[3] else flag[1]
            label += _label + str(_value) + ", "

        label += ")"
        return label

    # extract points from Yolo results
    def generate_yolo_points(res_df, confidence_threshold=0., colour=(255,999,0), thickness=3):
        if res_df.shape[0]:
            for _, row in res_df.iterrows():
                if row['confidence'] > confidence_threshold:
                    x0,y0,x1,y1 = list(map(lambda x: int(x), [row['xmin'], row['ymin'], row['xmax'], row['ymax']]))
                    yield row['name'], row['confidence'], x0,y0,x1,y1, colour, thickness
# opencv helpers ------------------------------------------------------------------------->>
