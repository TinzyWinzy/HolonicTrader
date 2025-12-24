import os
import tensorflow as tf
try:
    import intel_extension_for_tensorflow as itex
    print("ITEX loaded successfully.")
except ImportError:
    print("ITEX NOT found.")

try:
    import openvino as ov
    print("OpenVINO loaded successfully. Version:", ov.get_version())
except ImportError:
    print("OpenVINO NOT found.")

print("TensorFlow Version:", tf.__version__)
print("Physical Devices (All):", tf.config.list_physical_devices())
print("Physical Devices (XPU):", tf.config.list_physical_devices('XPU'))

if len(tf.config.list_physical_devices('XPU')) > 0:
    print("✅ Intel GPU (XPU) detected and ready for acceleration.")
else:
    print("❌ Intel GPU (XPU) NOT detected.")
