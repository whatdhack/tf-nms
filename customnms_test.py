# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for matmul ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

import tensorflow as tf
import customnms_ops
import time
import argparse


class CustomNmsTest(test.TestCase):

  @test_util.run_gpu_only
  def testCustomNms(self):
    with self.test_session():
      with ops.device("/gpu:0"):
        boxes = np.array([[1., 2., 3., 4.], [4., 5., 6., 7.], [7., 8., 9, 10.], [1., 2., 3., 4.], [4., 5., 6., 7.], [7., 8., 9, 10.], [1., 2., 3., 4.], [4., 5., 6., 7.], [7., 8., 9, 10.], ], dtype=np.float32 )
        scores = np.array ( [.6, .7, .8, .6, .7, .8, .6, .7, .8])
        max_output_size = len(boxes) 
        iou_threshold = 0.9
        #out =  customnms_ops.custom_nms_v2( boxes, scores, max_output_size, iou_threshold) 
        out =  customnms_ops.custom_nms_basic( boxes, scores, max_output_size, iou_threshold) 
        print ("test out: ", out)
        self.assertAllClose( out, 
            #np.array([[2, 4], [6, 8]]))
            #np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            np.array([0, 1, 0]) )


def test_run(algo):

    algos = {
            "v2":  customnms_ops.custom_nms_v2,
            "basic":  customnms_ops.custom_nms_basic
            }
    boxesi = tf.placeholder(tf.float32, shape=(None, 4), name="boxes")
    scoresi = tf.placeholder(tf.float32, shape=(None,), name="scores")
    max_output_sizei = tf.placeholder(tf.int32, shape=(), name="max_output_size")
    iou_thresholdi = tf.placeholder(tf.float32, shape=(), name="iou_threshold")
    output  = algos[algo](boxesi, scoresi, max_output_sizei, iou_thresholdi)

    independent_boxes = 48
    total_boxes = 54000
    repeat_count =  int (total_boxes/independent_boxes)
    boxes = repeat_count*[ [ j  for j in range(3*i+1,(3*i+1)+4)] for i in range(0,independent_boxes)]
    boxes = np.array(boxes, dtype=np.float32)
    scores = repeat_count*[ 0.6+0.1*i for i in range(0,independent_boxes)]
    scores = np.array (scores, dtype=np.float32)
    max_output_size = 100
    iou_threshold = 0.5
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # warm up
        for i in range (1,11):
            output_indices = sess.run([output], feed_dict = {boxesi:boxes,scoresi:scores, max_output_sizei:max_output_size, iou_thresholdi:iou_threshold }) 
        # real run
        time0 = time.time()
        for i in range (1, 101):
            time00 = time.time()
            output_indices = sess.run([output], feed_dict = {boxesi:boxes,scoresi:scores, max_output_sizei:max_output_size, iou_thresholdi:iou_threshold }) 
            print ( "time taken: ", (time.time() - time00), i)
        print ( "num boxes: ", scores.shape)
        print ( "time taken: ", (time.time() - time0)/(i))
    print ( "output_indices: ", output_indices)

if __name__ == '__main__':
    parser =  argparse.ArgumentParser(description='Run NMS algorithm in Tensorflow.')
    parser.add_argument ( "--algo", default="v2", help="An NMS algorithm from: v2, basic")
    args = parser.parse_args()

    #test.main()
    test_run(args.algo)
