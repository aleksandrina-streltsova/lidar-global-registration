#include <pcl/features/shot_lrf.h>

#include "shot_debug.h"

#define PST_PI 3.1415926535897932384626433832795
#define PST_RAD_45 0.78539816339744830961566084581988
#define PST_RAD_90 1.5707963267948966192313216916398
#define PST_RAD_135 2.3561944901923449288469825374596
#define PST_RAD_180 PST_PI
#define PST_RAD_360 6.283185307179586476925286766558
#define PST_RAD_PI_7_8 2.7488935718910690836548129603691

const double zeroDoubleEps15 = 1E-15;
const float zeroFloatEps8 = 1E-8f;

inline bool areEquals(double val1, double val2, double zeroDoubleEps = zeroDoubleEps15) {
    return (std::abs(val1 - val2) < zeroDoubleEps);
}

inline bool areEquals(float val1, float val2, float zeroFloatEps = zeroFloatEps8) {
    return (std::fabs(val1 - val2) < zeroFloatEps);
}

void SHOTEstimationDebug::setInputCloud(const PointCloudConstPtr &cloud) {
    this->input_ = cloud;
    volumes_.resize(cloud->size(), std::array<int, 32>{0});
}

void SHOTEstimationDebug::interpolateSingleChannelDebug(const pcl::Indices &indices,
                                                        const std::vector<float> &sqr_dists,
                                                        const int index,
                                                        std::vector<double> &binDistance,
                                                        const int nr_bins,
                                                        Eigen::VectorXf &shot) {
    const Eigen::Vector4f &central_point = (*this->input_)[(*this->indices_)[index]].getVector4fMap();
    const pcl::ReferenceFrame &current_frame = (*this->frames_)[index];

    Eigen::Vector4f current_frame_x(current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
    Eigen::Vector4f current_frame_y(current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
    Eigen::Vector4f current_frame_z(current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

    for (std::size_t i_idx = 0; i_idx < indices.size(); ++i_idx) {
        if (!std::isfinite(binDistance[i_idx]))
            continue;

        Eigen::Vector4f delta = (*this->surface_)[indices[i_idx]].getVector4fMap() - central_point;
        delta[3] = 0;

        // Compute the Euclidean norm
        double distance = sqrt(sqr_dists[i_idx]);

        if (areEquals(distance, 0.0))
            continue;

        double xInFeatRef = delta.dot(current_frame_x);
        double yInFeatRef = delta.dot(current_frame_y);
        double zInFeatRef = delta.dot(current_frame_z);

        // To avoid numerical problems afterwards
        if (std::abs(yInFeatRef) < 1E-30)
            yInFeatRef = 0;
        if (std::abs(xInFeatRef) < 1E-30)
            xInFeatRef = 0;
        if (std::abs(zInFeatRef) < 1E-30)
            zInFeatRef = 0;


        unsigned char bit4 = ((yInFeatRef > 0) || ((yInFeatRef == 0.0) && (xInFeatRef < 0))) ? 1 : 0;
        unsigned char bit3 = static_cast<unsigned char> (((xInFeatRef > 0) || ((xInFeatRef == 0.0) && (yInFeatRef > 0)))
                                                         ? !bit4 : bit4);

        assert (bit3 == 0 || bit3 == 1);

        int desc_index = (bit4 << 3) + (bit3 << 2);

        desc_index = desc_index << 1;

        if ((xInFeatRef * yInFeatRef > 0) || (xInFeatRef == 0.0))
            desc_index += (std::abs(xInFeatRef) >= std::abs(yInFeatRef)) ? 0 : 4;
        else
            desc_index += (std::abs(xInFeatRef) > std::abs(yInFeatRef)) ? 4 : 0;

        desc_index += zInFeatRef > 0 ? 1 : 0;

        // 2 RADII
        desc_index += (distance > this->radius1_2_) ? 2 : 0;
        this->volumes_[index][desc_index]++;

        int step_index = static_cast<int>(std::floor(binDistance[i_idx] + 0.5));
        int volume_index = desc_index * (nr_bins + 1);

        //Interpolation on the cosine (adjacent bins in the histogram)
        binDistance[i_idx] -= step_index;
        double intWeight = (1 - std::abs(binDistance[i_idx]));

        if (binDistance[i_idx] > 0)
            shot[volume_index + ((step_index + 1) % nr_bins)] += static_cast<float> (binDistance[i_idx]);
        else
            shot[volume_index + ((step_index - 1 + nr_bins) % nr_bins)] += -static_cast<float> (binDistance[i_idx]);

        //Interpolation on the distance (adjacent husks)

        if (distance > this->radius1_2_)   //external sphere
        {
            double radiusDistance = (distance - this->radius3_4_) / this->radius1_2_;

            if (distance > radius3_4_) //most external sector, votes only for itself
                intWeight += 1 - radiusDistance;  //peso=1-d
            else  //3/4 of radius, votes also for the internal sphere
            {
                intWeight += 1 + radiusDistance;
                shot[(desc_index - 2) * (nr_bins + 1) + step_index] -= static_cast<float> (radiusDistance);
            }
        } else    //internal sphere
        {
            double radiusDistance = (distance - this->radius1_4_) / this->radius1_2_;

            if (distance < this->radius1_4_) //most internal sector, votes only for itself
                intWeight += 1 + radiusDistance;  //weight=1-d
            else  //3/4 of radius, votes also for the external sphere
            {
                intWeight += 1 - radiusDistance;
                shot[(desc_index + 2) * (nr_bins + 1) + step_index] += static_cast<float> (radiusDistance);
            }
        }

        //Interpolation on the inclination (adjacent vertical volumes)
        double inclinationCos = zInFeatRef / distance;
        if (inclinationCos < -1.0)
            inclinationCos = -1.0;
        if (inclinationCos > 1.0)
            inclinationCos = 1.0;

        double inclination = std::acos(inclinationCos);

        assert (inclination >= 0.0 && inclination <= PST_RAD_180);

        if (inclination > PST_RAD_90 || (std::abs(inclination - PST_RAD_90) < 1e-30 && zInFeatRef <= 0)) {
            double inclinationDistance = (inclination - PST_RAD_135) / PST_RAD_90;
            if (inclination > PST_RAD_135)
                intWeight += 1 - inclinationDistance;
            else {
                intWeight += 1 + inclinationDistance;
                assert ((desc_index + 1) * (nr_bins + 1) + step_index >= 0 &&
                        (desc_index + 1) * (nr_bins + 1) + step_index < descLength_);
                shot[(desc_index + 1) * (nr_bins + 1) + step_index] -= static_cast<float> (inclinationDistance);
            }
        } else {
            double inclinationDistance = (inclination - PST_RAD_45) / PST_RAD_90;
            if (inclination < PST_RAD_45)
                intWeight += 1 + inclinationDistance;
            else {
                intWeight += 1 - inclinationDistance;
                assert ((desc_index - 1) * (nr_bins + 1) + step_index >= 0 &&
                        (desc_index - 1) * (nr_bins + 1) + step_index < descLength_);
                shot[(desc_index - 1) * (nr_bins + 1) + step_index] += static_cast<float> (inclinationDistance);
            }
        }

        if (yInFeatRef != 0.0 || xInFeatRef != 0.0) {
            //Interpolation on the azimuth (adjacent horizontal volumes)
            double azimuth = std::atan2(yInFeatRef, xInFeatRef);

            int sel = desc_index >> 2;
            double angularSectorSpan = PST_RAD_45;
            double angularSectorStart = -PST_RAD_PI_7_8;

            double azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan * sel)) / angularSectorSpan;

            assert ((azimuthDistance < 0.5 || areEquals(azimuthDistance, 0.5)) &&
                    (azimuthDistance > -0.5 || areEquals(azimuthDistance, -0.5)));

            azimuthDistance = (std::max)(-0.5, std::min(azimuthDistance, 0.5));

            if (azimuthDistance > 0) {
                intWeight += 1 - azimuthDistance;
                int interp_index = (desc_index + 4) % this->maxAngularSectors_;
                assert (interp_index * (nr_bins + 1) + step_index >= 0 &&
                        interp_index * (nr_bins + 1) + step_index < descLength_);
                shot[interp_index * (nr_bins + 1) + step_index] += static_cast<float> (azimuthDistance);
            } else {
                int interp_index = (desc_index - 4 + this->maxAngularSectors_) % this->maxAngularSectors_;
                assert (interp_index * (nr_bins + 1) + step_index >= 0 &&
                        interp_index * (nr_bins + 1) + step_index < descLength_);
                intWeight += 1 + azimuthDistance;
                shot[interp_index * (nr_bins + 1) + step_index] -= static_cast<float> (azimuthDistance);
            }

        }

        assert (volume_index + step_index >= 0 && volume_index + step_index < descLength_);
        shot[volume_index + step_index] += static_cast<float> (intWeight);
    }
}

void SHOTEstimationDebug::computePointSHOT(
        const int index, const pcl::Indices &indices, const std::vector<float> &sqr_dists, Eigen::VectorXf &shot) {
    //Skip the current feature if the number of its neighbors is not sufficient for its description
    if (indices.size() < 5) {
        PCL_WARN (
                "[pcl::%s::computePointSHOT] Warning! Neighborhood has less than 5 vertexes. Aborting description of point with index %d\n",
                getClassName().c_str(), (*this->indices_)[index]);

        shot.setConstant(this->descLength_, 1, std::numeric_limits<float>::quiet_NaN());

        return;
    }

    // Clear the resultant shot
    std::vector<double> binDistanceShape;
    this->createBinDistanceShape(index, indices, binDistanceShape);

    // Interpolate
    shot.setZero();
    interpolateSingleChannelDebug(indices, sqr_dists, index, binDistanceShape, this->nr_shape_bins_, shot);

    // Normalize the final histogram
    this->normalizeHistogram(shot, this->descLength_);
}