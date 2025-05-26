#include "mappoint.h"

MapPoint::MapPoint(long id, Vec3d position) : id_(id), pos_(position) {}

MapPointPtr MapPoint::CreateNewMappoint() {
    static long factory_id = 0;
    MapPointPtr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
    for (auto iter = observations_.begin(); iter != observations_.end();
         iter++) {
        if (iter->lock() == feat) {
            observations_.erase(iter);
            feat->map_point_.reset();
            observed_times_--;
            break;
        }
    }
}