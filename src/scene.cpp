#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;
using json = nlohmann::json;

// Dummy image loader callback for tinygltf (we don't use textures yet, just geometry)
static bool DummyLoadImageData(tinygltf::Image* image, const int image_idx, std::string* err,
                               std::string* warn, int req_width, int req_height,
                               const unsigned char* bytes, int size, void* user_data) {
    // We're not loading textures, just return success
    // Set minimal image data to avoid issues
    image->width = 1;
    image->height = 1;
    image->component = 4;
    image->bits = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    image->image.resize(4, 255);  // 1x1 white pixel
    return true;
}

// Helper to extract base path from a file path
static std::string getBasePath(const std::string& filepath) {
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return filepath.substr(0, lastSlash + 1);
    }
    return "";
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    
    basePath = getBasePath(filename);
    
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        
        // Load base color (required for all materials)
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        
        // Load PBR parameters with defaults
        newMaterial.metallic = p.contains("METALLIC") ? (float)p["METALLIC"] : 0.0f;
        newMaterial.roughness = p.contains("ROUGHNESS") ? (float)p["ROUGHNESS"] : 1.0f;
        newMaterial.transparency = p.contains("TRANSPARENCY") ? (float)p["TRANSPARENCY"] : 0.0f;
        newMaterial.indexOfRefraction = p.contains("IOR") ? (float)p["IOR"] : 1.5f;
        newMaterial.emittance = p.contains("EMITTANCE") ? (float)p["EMITTANCE"] : 0.0f;
        
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        newGeom.triangleStart = 0;
        newGeom.triangleCount = 0;
        newGeom.boundingBoxMin = glm::vec3(FLT_MAX);
        newGeom.boundingBoxMax = glm::vec3(-FLT_MAX);
        
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
        }
        else
        {
            newGeom.type = SPHERE;  // default
        }
        
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // Load mesh from glTF file if this is a mesh type
        if (newGeom.type == MESH && p.contains("FILE"))
        {
            std::string meshFile = basePath + std::string(p["FILE"]);
            int triangleStartIdx = static_cast<int>(triangles.size());
            
            loadGLTF(meshFile, newGeom.transform, newGeom.materialid);
            
            newGeom.triangleStart = triangleStartIdx;
            newGeom.triangleCount = static_cast<int>(triangles.size()) - triangleStartIdx;
            
            // Compute bounding box from loaded triangles
            for (int i = triangleStartIdx; i < static_cast<int>(triangles.size()); i++)
            {
                const Triangle& tri = triangles[i];
                newGeom.boundingBoxMin = glm::min(newGeom.boundingBoxMin, tri.v0);
                newGeom.boundingBoxMin = glm::min(newGeom.boundingBoxMin, tri.v1);
                newGeom.boundingBoxMin = glm::min(newGeom.boundingBoxMin, tri.v2);
                newGeom.boundingBoxMax = glm::max(newGeom.boundingBoxMax, tri.v0);
                newGeom.boundingBoxMax = glm::max(newGeom.boundingBoxMax, tri.v1);
                newGeom.boundingBoxMax = glm::max(newGeom.boundingBoxMax, tri.v2);
            }
            
            cout << "Loaded mesh with " << newGeom.triangleCount << " triangles" << endl;
            cout << "  Bounding box: [" << glm::to_string(newGeom.boundingBoxMin) 
                 << "] to [" << glm::to_string(newGeom.boundingBoxMax) << "]" << endl;
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    
    cout << "Scene loaded: " << geoms.size() << " geometry objects, " 
         << triangles.size() << " triangles total" << endl;
}

void Scene::loadGLTF(const std::string& gltfPath, const glm::mat4& transform, int materialId)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    
    // Set custom image loader (we don't use textures yet, just geometry)
    loader.SetImageLoader(DummyLoadImageData, nullptr);
    
    cout << "Loading glTF: " << gltfPath << endl;
    
    bool ret = false;
    // Check file extension to determine binary or ASCII format
    std::string ext = gltfPath.substr(gltfPath.find_last_of('.'));
    if (ext == ".glb")
    {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfPath);
    }
    else
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
    }
    
    if (!warn.empty())
    {
        cout << "glTF Warning: " << warn << endl;
    }
    
    if (!err.empty())
    {
        cerr << "glTF Error: " << err << endl;
    }
    
    if (!ret)
    {
        cerr << "Failed to load glTF: " << gltfPath << endl;
        return;
    }
    
    // Calculate normal matrix for transforming normals
    glm::mat3 normalMatrix = glm::mat3(glm::inverseTranspose(transform));
    
    // Process all meshes in the model
    for (const auto& mesh : model.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
            {
                cout << "  Skipping non-triangle primitive (mode=" << primitive.mode << ")" << endl;
                continue;
            }
            
            // Get position accessor
            const auto& posAccessorIt = primitive.attributes.find("POSITION");
            if (posAccessorIt == primitive.attributes.end())
            {
                cerr << "  Mesh primitive has no POSITION attribute" << endl;
                continue;
            }
            
            const tinygltf::Accessor& posAccessor = model.accessors[posAccessorIt->second];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            
            const float* positions = reinterpret_cast<const float*>(
                &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
            size_t posStride = posAccessor.ByteStride(posBufferView) / sizeof(float);
            if (posStride == 0) posStride = 3;  // tightly packed
            
            // Get normal accessor (optional)
            const float* normals = nullptr;
            size_t normStride = 3;
            const auto& normAccessorIt = primitive.attributes.find("NORMAL");
            if (normAccessorIt != primitive.attributes.end())
            {
                const tinygltf::Accessor& normAccessor = model.accessors[normAccessorIt->second];
                const tinygltf::BufferView& normBufferView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::Buffer& normBuffer = model.buffers[normBufferView.buffer];
                
                normals = reinterpret_cast<const float*>(
                    &normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
                normStride = normAccessor.ByteStride(normBufferView) / sizeof(float);
                if (normStride == 0) normStride = 3;
            }
            
            // Get indices
            if (primitive.indices < 0)
            {
                // Non-indexed geometry
                for (size_t i = 0; i < posAccessor.count; i += 3)
                {
                    Triangle tri;
                    
                    // Get vertex positions and transform to world space
                    for (int v = 0; v < 3; v++)
                    {
                        size_t idx = i + v;
                        glm::vec3 localPos(
                            positions[idx * posStride + 0],
                            positions[idx * posStride + 1],
                            positions[idx * posStride + 2]
                        );
                        glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                        
                        glm::vec3 normal(0, 1, 0);  // default up
                        if (normals)
                        {
                            normal = glm::vec3(
                                normals[idx * normStride + 0],
                                normals[idx * normStride + 1],
                                normals[idx * normStride + 2]
                            );
                            normal = glm::normalize(normalMatrix * normal);
                        }
                        
                        if (v == 0) { tri.v0 = worldPos; tri.n0 = normal; }
                        else if (v == 1) { tri.v1 = worldPos; tri.n1 = normal; }
                        else { tri.v2 = worldPos; tri.n2 = normal; }
                    }
                    
                    // Compute face normal if no vertex normals provided
                    if (!normals)
                    {
                        glm::vec3 faceNormal = glm::normalize(
                            glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                        tri.n0 = tri.n1 = tri.n2 = faceNormal;
                    }
                    
                    tri.materialId = materialId;
                    triangles.push_back(tri);
                }
            }
            else
            {
                // Indexed geometry
                const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                
                const void* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
                
                for (size_t i = 0; i < indexAccessor.count; i += 3)
                {
                    Triangle tri;
                    
                    // Get indices based on component type
                    uint32_t indices[3];
                    for (int v = 0; v < 3; v++)
                    {
                        size_t idx = i + v;
                        switch (indexAccessor.componentType)
                        {
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                                indices[v] = reinterpret_cast<const uint8_t*>(indexData)[idx];
                                break;
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                                indices[v] = reinterpret_cast<const uint16_t*>(indexData)[idx];
                                break;
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                                indices[v] = reinterpret_cast<const uint32_t*>(indexData)[idx];
                                break;
                            default:
                                indices[v] = 0;
                                break;
                        }
                    }
                    
                    // Get vertex positions and transform to world space
                    for (int v = 0; v < 3; v++)
                    {
                        uint32_t idx = indices[v];
                        glm::vec3 localPos(
                            positions[idx * posStride + 0],
                            positions[idx * posStride + 1],
                            positions[idx * posStride + 2]
                        );
                        glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                        
                        glm::vec3 normal(0, 1, 0);  // default up
                        if (normals)
                        {
                            normal = glm::vec3(
                                normals[idx * normStride + 0],
                                normals[idx * normStride + 1],
                                normals[idx * normStride + 2]
                            );
                            normal = glm::normalize(normalMatrix * normal);
                        }
                        
                        if (v == 0) { tri.v0 = worldPos; tri.n0 = normal; }
                        else if (v == 1) { tri.v1 = worldPos; tri.n1 = normal; }
                        else { tri.v2 = worldPos; tri.n2 = normal; }
                    }
                    
                    // Compute face normal if no vertex normals provided
                    if (!normals)
                    {
                        glm::vec3 faceNormal = glm::normalize(
                            glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                        tri.n0 = tri.n1 = tri.n2 = faceNormal;
                    }
                    
                    tri.materialId = materialId;
                    triangles.push_back(tri);
                }
            }
        }
    }
}
