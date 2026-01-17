#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// Triangle stored in world space for GPU ray intersection
struct Triangle
{
    glm::vec3 v0, v1, v2;    // Vertices
    glm::vec3 n0, n1, n2;    // Vertex normals (for smooth shading)
    int materialId;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    
    // For MESH type: indices into the global triangle array
    int triangleStart;
    int triangleCount;
    
    // Bounding box for mesh (in world space)
    glm::vec3 boundingBoxMin;
    glm::vec3 boundingBoxMax;
};

struct Material
{
    glm::vec3 color;          // Base color (albedo)
    float metallic;           // 0 = dielectric, 1 = metal
    float roughness;          // 0 = mirror, 1 = diffuse
    float transparency;       // 0 = opaque, 1 = fully transparent
    float indexOfRefraction;  // IOR for transparent materials (glass ~1.5, water ~1.33)
    float emittance;          // Light emission intensity (0 = not emissive)
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
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
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
  __host__ __device__ bool operator<(const ShadeableIntersection& other) const {
      return materialId < other.materialId; 
  }
};
