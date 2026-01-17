#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Schlick's Fresnel approximation
__host__ __device__ inline glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    return F0 + (1.0f - F0) * powf(fmaxf(1.0f - cosTheta, 0.0f), 5.0f);
}

// Sample GGX microfacet normal (importance sampling)
__host__ __device__ glm::vec3 sampleGGX(
    const glm::vec3& normal,
    float roughness,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float xi1 = u01(rng);
    float xi2 = u01(rng);
    
    float a = roughness * roughness;
    float a2 = a * a;
    
    // Sample spherical coordinates
    float phi = TWO_PI * xi1;
    float cosTheta = sqrtf((1.0f - xi2) / (1.0f + (a2 - 1.0f) * xi2));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    
    // Convert to Cartesian (local space)
    glm::vec3 H_local(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
    
    // Build tangent space basis
    glm::vec3 up = fabsf(normal.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    glm::vec3 tangent = glm::normalize(glm::cross(up, normal));
    glm::vec3 bitangent = glm::cross(normal, tangent);
    
    // Transform to world space
    return glm::normalize(tangent * H_local.x + bitangent * H_local.y + normal * H_local.z);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    if (pathSegment.remainingBounces <= 0) {
        return;
    }
    pathSegment.remainingBounces -= 1;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    
    // Emissive material - terminate and accumulate light
    if (m.emittance > 0.0f) {
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = 0;
        return;
    }
    
    // Ensure normal faces the ray
    glm::vec3 N = normal;
    bool entering = glm::dot(pathSegment.ray.direction, N) < 0.0f;
    if (!entering) {
        N = -N;
    }
    
    // Handle transparent/refractive materials
    if (m.transparency > 0.0f && u01(rng) < m.transparency) {
        float eta = entering ? (1.0f / m.indexOfRefraction) : m.indexOfRefraction;
        glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, N, eta);
        
        // Total internal reflection check
        if (glm::length(refractDir) < 0.001f) {
            // Total internal reflection - reflect instead
            glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, N);
            pathSegment.ray.origin = intersect + EPSILON * reflectDir;
            pathSegment.ray.direction = reflectDir;
        } else {
            pathSegment.ray.origin = intersect - EPSILON * N; // Go through surface
            pathSegment.ray.direction = glm::normalize(refractDir);
        }
        // Glass is typically colorless, but tinted glass would multiply by color
        pathSegment.color *= m.color;
        return;
    }
    
    // PBR metallic-roughness workflow
    // F0: reflectance at normal incidence
    // Dielectrics: ~0.04, Metals: base color
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, m.metallic);
    
    // View direction (pointing away from surface)
    glm::vec3 V = -pathSegment.ray.direction;
    float NdotV = fmaxf(glm::dot(N, V), 0.001f);
    
    // Fresnel at view angle - determines specular probability
    glm::vec3 fresnel = fresnelSchlick(NdotV, F0);
    float pSpecular = (fresnel.x + fresnel.y + fresnel.z) / 3.0f;
    
    // Clamp roughness to avoid division issues
    float roughness = fmaxf(m.roughness, 0.04f);
    
    if (u01(rng) < pSpecular) {
        // Specular reflection (GGX microfacet)
        glm::vec3 H = sampleGGX(N, roughness, rng);
        glm::vec3 newDir = glm::reflect(-V, H);
        
        // Ensure reflection is above surface
        if (glm::dot(newDir, N) <= 0.0f) {
            // Fallback to perfect reflection if sampled direction is invalid
            newDir = glm::reflect(pathSegment.ray.direction, N);
        }
        
        pathSegment.ray.origin = intersect + EPSILON * N;
        pathSegment.ray.direction = glm::normalize(newDir);
        
        // Specular color: metals use base color, dielectrics use white
        glm::vec3 specColor = glm::mix(glm::vec3(1.0f), m.color, m.metallic);
        pathSegment.color *= specColor * fresnel / fmaxf(pSpecular, 0.001f);
    }
    else {
        // Diffuse reflection (cosine-weighted hemisphere)
        glm::vec3 newDir = calculateRandomDirectionInHemisphere(N, rng);
        pathSegment.ray.origin = intersect + EPSILON * N;
        pathSegment.ray.direction = newDir;
        
        // Diffuse color: metals have no diffuse, dielectrics use base color
        glm::vec3 diffuseColor = m.color * (1.0f - m.metallic);
        pathSegment.color *= diffuseColor / fmaxf(1.0f - pSpecular, 0.001f);
    }
}
