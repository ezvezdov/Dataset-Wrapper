# Unificated categories list
    0:  "undefined"
    1:  "vehicle.car"
    2:  "vehicle.truck"
    3:  "vehicle.bicycle"
    4:  "vehicle.bus"
    5:  "vehicle.motorcycle"
    6:  "vehicle.emergency"
    7:  "vehicle.other"
    8:  "vehicle.ego"
    9:  "pedestrian"
    10: "animal"
    11: "vegetation"
    12: "road.drivable_surface"
    13: "road.non_drivable_surface"
    14: "building.obstacles"
    15: "building.other" 

# Description of categories
  * undefined - undefined objects.
  * vehicle.car - Vehicle designed primarily for personal use, e.g. sedans, hatch-backs, wagons, vans, mini-vans, SUVs and jeeps.
  * vehicle.truck - Vehicles primarily designed to haul cargo including pick-ups, lorrys, trucks and semi-tractors.
  * vehicle.bicycle - Human or electric powered 2-wheeled vehicle designed to travel at lower speeds either on road surface, sidewalks or bike paths.
  * vehicle.bus - Vehicle designed primarily for public transport (buses, trams e.g)
  * vehicle.motorcycle - Gasoline or electric powered 2-wheeled vehicle designed to move rapidly (at the speed of standard cars) on the road surface. This category includes all motorcycles, vespas and scooters.
  * vehicle.emergency - Vehicles of all types of emergency services.
  * vehicle.other - other possible vehicles or vehicles with undefined subcategory. 
  * vehicle.ego - The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image.
  * pedestrian - all types of human.
  * animal - All animals, e.g. cats, rats, dogs, deer, birds.
  * vegetation - Any vegetation in the frame that is higher than the ground, including bushes, plants, potted plants, trees, etc. Only tall grass (> 20cm) is part of this, ground level grass is part of road.non_drivable_surface.
  * road.drivable_surface - All paved or unpaved surfaces that a car can drive.
  * road.non_drivable_surface - All other forms of horizontal ground-level structures that do not belong to any of driveable_surface, curb, sidewalk and terrain. Includes elevated parts of traffic islands, delimiters, rail tracks, stairs with at most 3 steps and larger bodies of water (lakes, rivers).
  * building.obstacles - all man-made obstacles on the road
  * building.other - all man-made building near the road.


# Sorting dataset's categories into unificated categories list

Nuscenes dataset:

    undefined:
        noise
        static.other
    vehicle.car:
        vehicle.car
    vehicle.truck:
        vehicle.truck
    vehicle.bicycle:
        vehicle.bicycle
    vehicle.bus:
        vehicle.bus.bendy
        vehicle.bus.rigid
    vehicle.motorcycle:
        vehicle.motorcycle
    vehicle.emergency:
        vehicle.emergency.ambulance
        vehicle.emergency.police
    vehicle.other
        vehicle.construction
        vehicle.trailer
    vehicle.ego
        vehicle.ego
    pedestrian:
        human.pedestrian.adult
        human.pedestrian.child`
        human.pedestrian.construction_worker
        human.pedestrian.personal_mobility
        human.pedestrian.police_officer
        human.pedestrian.stroller
        human.pedestrian.wheelchair` 
    animal:
        animal
    vegetation:
        static.vegetation
    road.drivable_surface:
        flat.driveable_surface
    road.non_drivable_surface:
        flat.other
        flat.sidewalk
        flat.terrain
    building.obstacles:
        movable_object.barrier
        movable_object.debris
        movable_object.pushable_pullable
        movable_object.trafficcone
    building.other:
        static.manmade
        static_object.bicycle_rack
                        

Level 5 dataset 

    undefined:
    vehicle.car:
        car
    vehicle.truck:
        truck
    vehicle.bicycle:
        bicycle
    vehicle.bus:
        bus
    vehicle.motorcycle:
        motorcycle
    vehicle.emergency:
        emergency_vehicles
    vehicle.other:
        other_vehicle
    vehicle.ego
    pedestrian:
        pedestrian
    animal:
        animal
    vegetation:
    road.drivable_surface:
    road.non_drivable_surface:
    building.obstacles:
    building.other:
    


a2d2 dataset

    undefined:
        Sky
        Blurred area
        Sidebars
        Grid structure
    vehicle.car:
        Car 1
        Car 2
        Car 3
        Car 4
    vehicle.truck:
        Truck 1
        Truck 2
        Truck 3
    vehicle.bicycle:
        Bicycle 1
        Bicycle 2
        Bicycle 3
        Bicycle 4
    vehicle.bus:
    vehicle.motorcycle:
    vehicle.emergency:
    vehicle.other:
        Small vehicles 1
        Small vehicles 2
        Small vehicles 3
        Utility vehicle 1
        Utility vehicle 2
        Tractor
    vehicle.ego:
            Ego car
    pedestrian:
        Pedestrian 1
        Pedestrian 2
        Pedestrian 3
    animal:
        Animals
    vegetation:
        Nature object
    road.drivable_surface:
        Zebra crossing
        Slow drive area
        Dashed line
        RD normal street
        Parking area
        Speed bumper
        Painted driv. instr.
        Rain dirt
        Drivable cobblestone
    road.non_drivable_surface:
        Non-drivable street
        RD restricted area
        Sidewalk
        Solid line
    building.obstacles:
        Obstacles / trash
        Curbstone
        Road blocks
        Curbstone
    building.other:
        Poles
        Traffic signal 1
        Traffic signal 2
        Traffic signal 3
        Traffic sign 1
        Traffic sign 2
        Traffic sign 3  
        Irrelevant signs 
        Electronic traffic
        Traffic guide obj.
        Signal corpus
        Buildings


Waymo dataset

    undefined:
        UNDEFINED
    vehicle.car:
        CAR
    vehicle.truck:
        TRUCK
    vehicle.bicycle:
        BICYCLE
        BICYCLIST
    vehicle.bus:
    vehicle.motorcycle:
        MOTORCYCLIST
        MOTORCYCLE
    vehicle.emergency:
    vehicle.other:
        OTHER_VEHICLE
    vehicle.ego:
    pedestrian:
        PEDESTRIAN
    animal:
    vegetation:
        VEGETATION
        TREE_TRUNK
    road.drivable_surface:
        ROAD
        LANE_MARKER
    road.non_drivable_surface:
        WALKABLE
        OTHER_GROUND
        SIDEWALK
    building.obstacles:
        CURB
        CONSTRUCTION_CONE
    building.other:
        BUILDING
        POLE
        SIGN
        TRAFFIC_LIGHT

Waymo dataset bounding boxes

    undefined:
        TYPE_UNSET
        TYPE_OTHER
    vehicle.bicycle:
        TYPE_CYCLIST
    vehicle.other:
        TYPE_VEHICLE
    pedestrian:
        TYPE_PEDESTRIAN
    
