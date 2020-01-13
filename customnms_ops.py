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
"""Use matmul ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

print ( "custom_nms_ops.py")
customnms_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('custom_nms_ops.so'))
print ( "custom_nms_ops: ", dir(customnms_ops))
custom_nms_v2 = customnms_ops.custom_nms_v2
custom_nms_v3 = customnms_ops.custom_nms_v3
custom_nms_basic = customnms_ops.custom_nms_basic

