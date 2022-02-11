#ifndef REGISTRATION_IO_H
#define REGISTRATION_IO_H

#include <pcl/io/ply_io.h>

template<typename PointT>
inline int loadPLYFile(const std::string &file_name, pcl::PointCloud<PointT> &cloud,
                       std::vector<pcl::PCLPointField> &fields, const int offset = 0) {
    pcl::PCLPointCloud2 blob;
    int ply_version;
    pcl::PLYReader p;
    int res = p.read(file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_, ply_version, offset);

    // Exit in case of error
    if (res < 0)
        return res;
    fromPCLPointCloud2(blob, cloud);
    fields = blob.fields;
    return 0;
}

#endif
