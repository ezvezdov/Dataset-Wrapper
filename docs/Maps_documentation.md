# NuScenes and Lyft maps
get_map() method returns list of nuscenes.utils.map_mask.MapMask objects.

```python
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/map_mask.py
from nuscenes.utils.map_mask import MapMask
map_list = nusc.get_map()
map = map_list[0]
```

##### MapMask attributes:
```python
map.img_file  # path to map image
map.resolution # map image resolution
map.foreground = 255 #foregroung color
map.background = 0 #background color
```


##### MapMask methods:
```python
# Returns the map mask, optionally dilated.
map.mask(self, dilation: float = 0.0) -> np.ndarray

# Generate transform matrix for this map mask.
map.transform_matrix() -> np.ndarray

# Determine whether the given coordinates are on the (optionally dilated) map mask.
map.is_on_mask(x: Any, y: Any, dilation: float = 0) -> np.array

#Maps x, y location in global map coordinates to the map image coordinates.
map.to_pixel_coords(self, x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]
```