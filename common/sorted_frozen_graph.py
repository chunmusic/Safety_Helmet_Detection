import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.gfile.FastGFile('/home/pongsakorn/kaggle_ws/road_sign_ws/output/export_saved_model/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph_def = TransformGraph(graph_def, ['image_tensor'], ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections'], ['sort_by_execution_order'])
    with tf.gfile.FastGFile('/home/pongsakorn/kaggle_ws/road_sign_ws/sorted_inference_graph.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

