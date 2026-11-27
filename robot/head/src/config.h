#pragma once

// Wiring
#define PAN_PIN 18
#define TILT_PIN 19

// How far each servo may turn, in degrees (90 looks straight ahead)
#define PAN_MIN 10
#define PAN_MAX 170
#define TILT_MIN 45
#define TILT_MAX 135

// How fast the head turns, in degrees per second. Hobby servos manage about 500.
#define SPEED 240

// Network
#define HOSTNAME "machine-head"   // reachable as machine-head.local
#define PORT 7351

#ifndef WIFI_SSID
#if __has_include("secrets.h")
#include "secrets.h"
#endif
#endif
#ifndef WIFI_CHANNEL
#define WIFI_CHANNEL 0
#endif
