#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadGLTF(const std::string& gltfPath, const glm::mat4& transform, int materialId);
    std::string basePath;  // Directory containing the scene file
    
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;  // All triangles from all meshes
    RenderState state;
};
