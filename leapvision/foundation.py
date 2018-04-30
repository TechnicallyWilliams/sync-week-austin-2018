import rtree
import math


def box_opposite_corners(box):
    '''
    Takes a box in the form (x, y, width, height) and returns it as a tuple in
    the form of (x1, y1, x2, y2) where x1 < x2 and y1 < y2.
    '''
    return (box[0], box[1], box[0]+box[2], box[1]+box[3])


def box_intersection_area(a, b):
    '''
    Returns the area of intersection between rectangles a and b which are in
    the form (x1, y1, x2, y2) where x1 < x2 and y1 < y2. 
    '''
    minX = max(a[0], b[0])
    minY = max(a[1], b[1])
    maxX = min(a[2], b[2])
    maxY = min(a[3], b[3])
    return float(max(0, maxX - minX) * max(0, maxY - minY))


def box_union_area(a, b):
    '''
    Returns the area of union between rectangles a and b which are in
    the form (x1, y1, x2, y2) where x1 < x2 and y1 < y2. 
    '''
    if a[2] < b[0] or b[2] < a[0]:
        # No intersection in x
        return 0
    if a[3] < b[1] or b[3] < a[1]:
        # No intersection in y
        return 0
    return float((max(b[2], a[2]) - min(b[0], a[0])) * (max(b[3], a[3]) - min(b[1], a[1])))


def box_intersection_over_union(a, b):
    '''
    Takes the area of the intersecting rect of boxes a and b and divides it by
    their union rect. Rectangles should be expressed as (x1, y1, x2, y2).
    '''
    union = box_union_area(a, b)
    if union == 0:
        return 0.0
    intersection = box_intersection_area(a, b)
    return (intersection / float(union))


def non_max_suppression(objects, score_key='score', box_key='box', iou_threshold=0.5, should_convert_box=True):
    '''
    Filters the given list of objects using non-max-suppression. If the box
    value for each object is NOT in the form (x1, y1, x2, y2) but rather
    (x, y, width, height), then should_convert_box just be set to True.
    '''
    sorted_objects = sorted(objects, key=lambda x: x[score_key], reverse=True)

    database = rtree.index.Index()
    for (i, entry) in enumerate(sorted_objects):
        box = box_opposite_corners(
            entry[box_key]) if should_convert_box else entry[box_key]
        database.insert(i, box)

    indexes_to_ignore = set()
    indexes_to_return = set()
    for (i, entry) in enumerate(sorted_objects):
        if i in indexes_to_ignore:
            continue
        # The current object at index i should be added because it has the
        # highest confidence score compared to any that potentially intersect.
        indexes_to_return.add(i)
        a = box_opposite_corners(
            sorted_objects[i][box_key]) if should_convert_box else sorted_objects[i][box_key]
        xsect = database.intersection(a)
        for x in xsect:
            if x == i:
                # Ignore the current object since it will always return from
                # the intersection lookup
                continue
            b = box_opposite_corners(
                sorted_objects[x][box_key]) if should_convert_box else sorted_objects[x][box_key]
            iou = box_intersection_over_union(a, b)
            if iou > iou_threshold:
                indexes_to_ignore.add(x)

    return map(lambda x: sorted_objects[x], indexes_to_return)
