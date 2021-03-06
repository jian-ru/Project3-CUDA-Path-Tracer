#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define MAX_BVH_DEPTH 32
#define BVH_LEAF_SIZE 1
#define MIN_OBJECTS_REQUIRED_FOR_BVH 32

enum GeomType
{
    SPHERE,
    CUBE,
	MESH,
	GeomTypeCount
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle
{
	glm::vec3 positions[3];
	glm::vec3 normals[3];
	int meshIdx; // mesh this triangle belongs to
	int idx;
};

struct Geom
{
	int idx;
    enum GeomType type;
    int materialid;
	int triListOffset;
	int numTriangles;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum MaterialType
{
	M_Lambert,
	M_Mirror,
	M_Glass,
	M_AshikhminShirley,
	M_TorranceSparrow,
	M_MaterialCount
};

struct Material
{
    glm::vec3 color;
    struct
	{
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
	MaterialType type;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
	float lensRadius;
	float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec4> image;
    std::string imageName;
};

struct PathSegment
{
	Ray ray;
	glm::vec3 color;
	glm::vec3 misWeight;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
