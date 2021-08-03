from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import pdb

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

ERROR = 1e-6

class DetectionMetricsEstimatorTest(tf.test.TestCase):

    def get_boxes_from_bin(self, file):
        pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
        stuff1 = metrics_pb2.Objects()
        with open(file, 'rb')as rf:
            stuff1.ParseFromString(rf.read())
            for i in range(len(stuff1.objects)):
                obj = stuff1.objects[i].object
                pd_frame_id.append(stuff1.objects[i].frame_timestamp_micros)
                box = [obj.box.center_x, obj.box.center_y, obj.box.center_z,
                       obj.box.length, obj.box.width, obj.box.height, obj.box.heading]
                pd_bbox.append(box)
                pd_score.append(stuff1.objects[i].score)
                pd_type.append(obj.type)

                if obj.num_lidar_points_in_box and obj.num_lidar_points_in_box<=5:
                    difficulty.append(2)
                else:
                    difficulty.append(1)
        return np.array(pd_bbox), np.array(pd_type), np.array(pd_frame_id), np.array(pd_score), np.array(difficulty)

    def get_boxes_from_txt(self, pd_set, gt_set, pd_dir, gt_dir):
        __type_list = {'unknown': 0, 'Car': 1, 'Pedestrian': 2, 'Sign': 3, 'Cyclist': 4}
        pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = [], [], [], [], []
        gt_bbox, gt_type, gt_frame_id, gt_score, gt_diff = [], [], [], [], []
        f = open(gt_set, 'r')
        lines = f.readlines()
        f.close()
        #import pdb; pdb.set_trace()
        f = open(pd_set, 'r')
        pred_lines = f.readlines()
        f.close() 
        for i in range(39848):
            print('Current index:', i)
            gt_seg, gt_id = lines[i].strip().split(' ')
            gt_file_name = os.path.join(gt_dir , 'waymo2kitti', 'validation', gt_seg, 'label_0', gt_id + '.txt')
            
            file_name = pred_lines[i].strip() + '.txt' #str('{0:06}'.format(i)) + '.txt'
            file = os.path.join(pd_dir, file_name)
            if not os.path.exists(file):
                continue
        #    import pdb; pdb.set_trace()
            with open(file, 'r')as f:
                for line in f.readlines():
                    line = line.strip('\n').split()
                    #if float(line[15])==0:
                    #    continue
                    pd_frame_id.append(gt_id)
                    box = [float(line[11]), float(line[12]), float(line[13]),
                           float(line[10]), float(line[9]), float(line[8]),float(line[14])]
                    pd_bbox.append(box)
                    pd_score.append(line[15])
                    
                    pd_type.append(__type_list[line[0]])
                    
                    difficulty.append(1)
                   
            #import pdb; pdb.set_trace()

            with open(gt_file_name, 'r')as f:
                for line in f.readlines():
                    line = line.strip('\n').split()
                    if line[0]!= 'Car' or float(line[15])==0 or (float(line[4])-float(line[6]))>=0 or (float(line[5])-float(line[7]))>=0:
                        # print('=========ignore', line[0], line[15], line[4:8])
                        continue
                    gt_frame_id.append(gt_id)
                    box = [float(line[11]), float(line[12]), float(line[13]),
                           float(line[10]), float(line[9]), float(line[8]),float(line[14])]
                    gt_bbox.append(box)
                    gt_score.append('0.5')
                    # else: # gt
                    #     pd_score.append(0.5)
                    gt_type.append(__type_list[line[0]])
                    if float(line[15])>5:
                        gt_diff.append(1)
                    else:
                        gt_diff.append(2)
    #        import pdb; pdb.set_trace()

        return np.array(pd_bbox), np.array(pd_type), np.array(pd_frame_id), np.array(pd_score), np.array(difficulty),np.array(gt_bbox), np.array(gt_type), np.array(gt_frame_id), np.array(gt_score), np.array(gt_diff)

    def _BuildConfig(self):
        config = metrics_pb2.Config()
        # pdb.set_trace()
        config_text = """
    num_desired_score_cutoffs: 11
    breakdown_generator_ids: OBJECT_TYPE
    breakdown_generator_ids: RANGE
    difficulties {
    levels: 1
    levels: 2
    }
    difficulties {
    levels: 1
    levels: 2
    }
    matcher_type: TYPE_HUNGARIAN
    iou_thresholds: 0.0
    iou_thresholds: 0.7
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    iou_thresholds: 0.5
    box_type: TYPE_3D
    """
        text_format.Merge(config_text, config)
        return config

    def _BuildGraph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self._BuildConfig(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=tf.zeros_like(
                    self._pd_frame_id, dtype=tf.bool),
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                # ground_truth_difficulty=tf.ones_like(self._gt_frame_id, dtype=tf.uint8),
                ground_truth_difficulty=self._gt_difficulty,
                recall_at_precision=0.95,
            )
            return metrics

    def _EvalUpdateOps(
            self,
            sess,
            graph,
            metrics,
            prediction_frame_id,
            prediction_bbox,
            prediction_type,
            prediction_score,
            ground_truth_frame_id,
            ground_truth_bbox,
            ground_truth_type,
            ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        ddd = {}
        for item in metrics.items():
            #import pdb; pdb.set_trace()
            ddd[item[0]] = sess.run([item[1][0]])
        return ddd

    def testAPBasic(self):
        print("start")
        print(pd_set)
        print(gt_set)
        
        pd_bbox, pd_type, pd_frame_id, pd_score, _, gt_bbox, gt_type, gt_frame_id, _, difficulty = self.get_boxes_from_txt(pd_set, gt_set, pd_dir, gt_dir)
        # import pdb; pdb.set_trace()
        # pd_bbox, pd_type, pd_frame_id, pd_score, difficulty = self.get_boxes_from_bin(pd_file)
        # gt_bbox, gt_type, gt_frame_id = pd_bbox, pd_type, pd_frame_id
        graph = tf.Graph()
        metrics = self._BuildGraph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            #import pdb; pdb.set_trace()
            self._EvalUpdateOps(sess, graph, metrics, pd_frame_id, pd_bbox, pd_type,
                                pd_score, gt_frame_id, gt_bbox, gt_type, difficulty)

            aps = self._EvalValueOps(sess, graph, metrics)
            for key, value in aps.items():
                print(key, ":", value)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # pd_set and gt_set are the validation set index, pd_set is converted to the index pattern from gt_set using set_split.py
    pd_set = '/home/ubuntu/drive2/code/pct/data/val.txt'
    gt_set = '/home/ubuntu/drive2/code/pct/data/val_org.txt'
    # pd_path is path of the predicted data for val set, gt_path is the generated waymo label in kitti pattern
    pd_path = '/home/ubuntu/drive2/code/pct/experiments/pct/output/data'
    gt_path = '/home/ubuntu/drive2/code/waymo_kitti_converter/data'
    
    tf.compat.v1.disable_eager_execution()
    tf.test.main()

