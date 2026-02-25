# IMU 


The BMX160 can output angular velocity, acceleration, magnetic field, hall effect sensor readings, temperature and sensor time. Acceleration and magnetic field data are not used. We have some interest in storing and potentially sending down magnetic field data because it could potentially become a useful addition for the orbit determination problem. Temperature is useful because the gyro bias drifts with temperature, and even if we don't use it for orbit determination, we can assess on ground whether these bias variations are noticeable, and if considering it in the problem could lead to better accuracy.

## Gyro configuration

The BMX160 gyro outputs angular velocity in degrees per second. Data collection can be configured with three different parameters: Output Data Rate (ODR), Range and bandwidth. 

- Gyro range can be configured to a set of configurations between 125 dps and 2000 dps. The trade-off in choosing a higher range is lower resolution. If we were spinning at 125 dps we would be more worried about detumbling the satellite than running the experiment, so we can confidently set it by default at 125 dps to max out the resolution.
- Output data rate has a set of configurations between 25 Hz and 3200 Hz. The trade-off in choosing a higher vs lower one has to do with the bandwidth of the measurements.
- Bandwidth has three options: normal, over sampling by two and by 4. What this means is that the bandwidth of the sensor will be ~ half of the ODR in normal mode, 1/4 in OSR1 and 1/8 in OSR2. The lower the bandwidth, the less noise, provided the sensor is not also filtering out the dynamics we're hoping to capture. The spectral behavior of the gyro will be dominateed by the gyroscopic torque, which will have a frequency in the order of the angular velocity times some factors related to the anisotropy of the inertia matrix. The lowest bandwidth we can configure the gyro two is 2.5 Hz in OSR2 at 25 Hz, and the spacecraft would have to be spinning at that rate for it to start filtering the dynamics, but if we got to such a point we wouldn't be running the experiment anyways. 

## Temperature Readings

The BMX160 outputs die temperature. There are no configuration parameters for it. Has mentioned above, it could be useful to estimate gyro bias variations. It should be included in the payload telemetry, its possible the jetson may fail due to thermal considerations and any temperature data on the jetson carrier will be relevant for telemetry.

## Magnetometer Readings

The magnetometer also has no configuration parameters. Its range is set to cover the expected magnetic field strength on the earth's surface, which is close to what is seen in LEO. This is very secondary, so it's not worth going into much detail here. The data below is in microTesla.


## IMU Power Modes

There are four power modes: suspend, low power, fast start-up and normal. The gyro has no low power mode, and the magnetometer has no fast start-up mode. Suspend and fast start-up are considered sleep modes, and data can't be outputted in these.

## IMU Manager Modes

The IMU Manager has three modes: idle, collect and error. On idle, the gyro is set to fast start-up and the magnetometer is suspended. The accelerometer is always suspended. On collect, the gyro and magnetometer are set to normal. The error mode is meant for debugging/ trying to recover the imu in case an error flag is thrown.

## Error Handling

Functions were implemented to help debug in case errors are thrown by the sensor, although for the moment no error handling has been implemented. The bmx160 can throw error flags has defined in the datasheet. The bmx160.cpp and IMU Manager have functions to read these registries, although these are not meant to be used during nominal operations. If there is time, this can be handled better in the future.