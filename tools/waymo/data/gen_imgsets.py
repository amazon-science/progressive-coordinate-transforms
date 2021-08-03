import os

root_dir = 'data/waymo2kitti/training/'
seg_list = sorted(os.listdir(root_dir))

interval = 3

f = open('train_org.txt', 'w')
for seg in seg_list:
    if 'segment' in seg:
        image_list = os.listdir(root_dir + seg+'/label_0')
        img_list = sorted(image_list, key=lambda c: int(c.split('.')[0]))
#        import pdb; pdb.set_trace()
        for image in img_list:
            if int(float(image.split('.')[0])) % interval == 0:
                f.write(seg + ' ' + image.split('.')[0] + '\n')

f.close()
