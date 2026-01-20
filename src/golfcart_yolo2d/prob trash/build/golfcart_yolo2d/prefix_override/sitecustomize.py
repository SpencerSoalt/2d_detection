import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/ws/src/golfcart_yolo2d/install/golfcart_yolo2d'
