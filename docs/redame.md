Waymo dataset: some bounding boxes have category "vehicle.other", it means all types of vehicles (excluding bicycles),
because of subcategories aren't in waymo bounding boxes dataset

# Bboxes orientation

The yaw angle in radians of the forward direction of the bounding box (the
vector from the center of the box to the middle of the front box segment)
counter clockwise from the X-axis (right hand system about the Z axis).
This angle is normalized to [-pi, pi).