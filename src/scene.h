#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& gltfPath, const glm::mat4& transform, int materialId);
    void buildBVH();  // Build BVH for all meshes
    void buildMeshBVH(int geomIndex);  // Build BVH for a single mesh
    std::string basePath;  // Directory containing the scene file
    
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;  // All triangles from all meshes (reordered for BVH)
    std::vector<BVHNode> bvhNodes;    // All BVH nodes for all meshes
    std::vector<BVHInfo> bvhInfos;    // Per-mesh BVH metadata
    RenderState state;
};
