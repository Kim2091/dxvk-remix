/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

// Particle volume compute pass binding indices

// Constants
#define PARTICLE_VOLUME_BINDING_CONSTANTS                  0

// 3D volume textures (read-only inputs)
#define PARTICLE_VOLUME_BINDING_DENSITY_INPUT              1
#define PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT          2
#define PARTICLE_VOLUME_BINDING_VELOCITY_INPUT             3
#define PARTICLE_VOLUME_BINDING_OBSTACLE_INPUT             4
#define PARTICLE_VOLUME_BINDING_PRESSURE_INPUT             5
#define PARTICLE_VOLUME_BINDING_PREV_VELOCITY_INPUT        6

// 3D volume textures (read-write outputs)
#define PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT             10
#define PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT         11
#define PARTICLE_VOLUME_BINDING_VELOCITY_OUTPUT            12
#define PARTICLE_VOLUME_BINDING_OBSTACLE_OUTPUT            13
#define PARTICLE_VOLUME_BINDING_PRESSURE_OUTPUT            14

// Particle buffer (for splatting pass)
#define PARTICLE_VOLUME_BINDING_PARTICLES_INPUT            20

// Depth buffer / world position (for obstacle rasterization)
#define PARTICLE_VOLUME_BINDING_PREV_WORLD_POSITION_INPUT  21
#define PARTICLE_VOLUME_BINDING_DEPTH_INPUT                22

// Blackbody LUT
#define PARTICLE_VOLUME_BINDING_BLACKBODY_LUT_INPUT        23

// ReSTIR light output
#define PARTICLE_VOLUME_BINDING_LIGHT_OUTPUT               24

// Froxel grid output
#define PARTICLE_VOLUME_BINDING_FROXEL_DENSITY_OUTPUT      25
#define PARTICLE_VOLUME_BINDING_FROXEL_EMISSION_OUTPUT     26
